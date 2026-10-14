"""Contract tests using synthetic state, no GPU, model account or real navigation."""
import asyncio
import copy
import io
import math
import sys
import time
import unittest
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mhs"))
from nav_device import NavDevice, map_pose, target_distance
from agnes_mhs import CallError
from agnes_mhs.schema import errors
from navlive.navlive_server import Session
from PIL import Image


def state():
    return {"scene_id": "test-scene", "session_id": "owned-test", "seq": 0,
            "pose": {"position": [2., 0., 3.], "rotation": [1., 0., 0., 0.]},
            "target": [2., 0., 5.], "target_label": "厨房的桌子", "step_count": 0,
            "plan": [], "paused": True, "events_tail": [],
            "metrics": {"collisions": 0, "distance_to_goal": 99},
            "occupancy": {"bounds": [-10,-10,10,10], "resolution": .05, "origin": [-10,-10]}}


class Client:
    def __init__(self, outcome="arrived"):
        self.snapshot = state()
        self.outcome = outcome
        self.stopped = False
        self.started = False
        self.leases = 0

    async def state(self):
        if self.started:
            self.snapshot["seq"] += 1
            if self.outcome == "arrived":
                self.snapshot["pose"]["position"] = [2., 0., 5.]
                self.snapshot["plan"] = []
                self.snapshot["events_tail"] = [{"seq": self.snapshot["seq"], "kind": "waypoint_reached"}]
            elif self.outcome == "stalled":
                self.snapshot["events_tail"] = [{"seq": self.snapshot["seq"], "kind": "stalled"}]
            elif self.outcome == "silent":
                self.snapshot["plan"] = []
        return copy.deepcopy(self.snapshot)

    async def action(self, kind, **kwargs):
        if kind == "goto_landmark":
            self.started = True
            self.snapshot["plan"] = [{"label": "厨房的桌子"}]
            self.snapshot["paused"] = False
        if kind == "resume":
            self.snapshot["paused"] = False
        return copy.deepcopy(self.snapshot)

    async def lease(self):
        self.leases += 1

    async def stop(self):
        self.stopped = True
        self.snapshot["paused"] = True
        return copy.deepcopy(self.snapshot)


class Call:
    paused = False
    async def checkpoint(self, rest):
        await asyncio.sleep(0)
    def progress(self, **kwargs):
        self.progress_data = kwargs


def bridge(client):
    buf = io.BytesIO()
    Image.new("RGB", (32, 24)).save(buf, format="JPEG")
    return NavDevice(client, state(), [{"name": "厨房的桌子", "position": [2.,0.,5.]}], buf.getvalue())


class Contract(unittest.IsolatedAsyncioTestCase):
    def test_registration_matches_official_schema(self):
        device = bridge(Client())
        self.assertEqual(list(errors("RegisterParams", device.dev.description())), [])
        self.assertEqual(device.map["places"][0]["at"], [2., -5.])

    def test_pose_uses_camera_heading_and_metre_axes(self):
        s = state()
        self.assertEqual(map_pose(s, "floor")["y"], -3.)
        self.assertAlmostEqual(map_pose(s, "floor")["yaw"], 90.)
        s["pose"]["rotation"] = [math.cos(math.pi/4), 0, math.sin(math.pi/4), 0]
        self.assertAlmostEqual(map_pose(s, "floor")["yaw"], 180.)
        s["pose"]["rotation"] = [-math.cos(math.pi/4), 0, -math.sin(math.pi/4), 0]
        self.assertAlmostEqual(map_pose(s, "floor")["yaw"], 180.)

    def test_distance_is_current_target_not_benchmark_goal(self):
        self.assertEqual(target_distance(state()), 2.)

    def test_other_scene_cannot_be_shown_on_current_map(self):
        d = bridge(Client())
        other = state()
        other["scene_id"] = "another-scene"
        with self.assertRaises(ValueError):
            d.update(other)
        self.assertEqual(d.snapshot["scene_id"], "test-scene")

    async def test_arrival_requires_events_and_stops(self):
        c = Client()
        result = await bridge(c).navigate(Call(), ["厨房的桌子"])
        self.assertIn("已依次到达", result["detail"])
        self.assertTrue(c.stopped)

    async def test_empty_plan_is_not_success(self):
        c = Client("silent")
        with self.assertRaises(CallError) as raised:
            await bridge(c).navigate(Call(), ["厨房的桌子"])
        self.assertEqual(raised.exception.reason, "x_incomplete")
        self.assertTrue(c.stopped)

    async def test_stall_returns_real_failure_and_stops(self):
        c = Client("stalled")
        with self.assertRaises(CallError) as raised:
            await bridge(c).navigate(Call(), ["厨房的桌子"])
        self.assertEqual(raised.exception.reason, "x_stalled")
        self.assertTrue(c.stopped)

    async def test_unknown_place_never_moves(self):
        c = Client()
        with self.assertRaises(CallError):
            await bridge(c).navigate(Call(), ["游泳池"])
        self.assertFalse(c.started)

    async def test_cancel_always_stops_and_heartbeat_exits(self):
        c = Client("waiting")
        task = asyncio.create_task(bridge(c).navigate(Call(), ["厨房的桌子"]))
        await asyncio.sleep(.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(c.stopped)
        n = c.leases
        await asyncio.sleep(.55)
        self.assertEqual(c.leases, n)

    async def test_navigation_timeout_stops(self):
        c = Client("waiting")
        device = bridge(c)
        device.run_timeout = -.1
        with self.assertRaises(CallError) as raised:
            await device.navigate(Call(), ["厨房的桌子"])
        self.assertEqual(raised.exception.reason, "x_navigation_timeout")
        self.assertTrue(c.stopped)

    async def test_pause_keeps_lease_until_resume(self):
        c = Client("waiting")
        device = bridge(c)
        reached_pause = asyncio.Event()
        resumed = asyncio.Event()
        class PausedCall(Call):
            paused = True
            async def checkpoint(self, rest):
                await rest()
                reached_pause.set()
                await resumed.wait()
                c.outcome = "arrived"
                self.paused = False
        task = asyncio.create_task(device.navigate(PausedCall(), ["厨房的桌子"]))
        await reached_pause.wait()
        before = c.leases
        await asyncio.sleep(.65)
        self.assertGreater(c.leases, before)
        resumed.set()
        await task
        self.assertTrue(c.stopped)


class Watchdog(unittest.TestCase):
    def session(self):
        s = Session.__new__(Session)
        s._control_lease = None
        s.paused = False
        s.events = []
        s._seq = 0
        return s

    def test_old_clients_without_lease_are_unchanged(self):
        s = self.session()
        s.check_control_lease()
        self.assertFalse(s.paused)

    def test_sim_process_stops_when_bridge_disappears(self):
        s = self.session()
        s.renew_control_lease("test-owner", 4)
        s._control_lease["deadline"] = time.monotonic() - 1
        s.check_control_lease()
        self.assertTrue(s.paused)
        self.assertEqual(s.events[-1]["kind"], "control_lease_expired")
        n = len(s.events)
        s.check_control_lease()
        self.assertEqual(len(s.events), n)

    def test_lease_cannot_be_taken_over(self):
        s = self.session()
        s.renew_control_lease("first", 4)
        with self.assertRaises(ValueError):
            s.renew_control_lease("other", 4)
        with self.assertRaises(ValueError):
            s.renew_control_lease("first", 60)

    def test_native_grid_uses_navmesh_and_matches_mhs_axes(self):
        import numpy as np
        class Pathfinder:
            def get_bounds(self):
                return [2., 0., 3.], [2.1, 1., 3.1]
            def get_topdown_view(self, resolution, height):
                return np.array([[True, False], [False, True]])
        s = self.session()
        s._grid_png = None
        s._pose = {"position": [2.,0.,3.]}
        s.env = SimpleNamespace(sim=SimpleNamespace(pathfinder=Pathfinder()))
        s.build_occupancy()
        self.assertEqual(s._grid_meta["origin"], [2., -3.1])
        self.assertEqual(s._grid_meta["bounds"], [2.,-3.1,2.1,-3.])
        with Image.open(io.BytesIO(s._grid_png)) as img:
            self.assertEqual(img.mode, "L")
            self.assertEqual(list(img.tobytes()), [255, 0, 0, 255])

    def test_free_tour_counts_real_steps_without_restarting_episode(self):
        s = self.session()
        s.free_tour = True
        s.episode_finished = False
        s.plan = [{"label": "desk", "position": [2.,0.,0.]}]
        s.step_count = 7
        s.trail_world = []
        s._stall_anchor = None
        s._stall_count = 0
        s.follower = SimpleNamespace(get_next_action=lambda target: 1)
        s.env = SimpleNamespace(step=lambda action: {"real-step": action})
        s.pose = lambda: {"position": [.25,0.,0.]}
        s._tick_metrics = lambda: None
        s._observe_calibration = lambda: None
        s.refresh_views = lambda: None
        s.tick()
        self.assertEqual(s.step_count, 8)
        self.assertEqual(s.trail_world, [[.25,0.,0.]])
        self.assertEqual(s.plan[0]["label"], "desk")

    def test_continuing_episode_preserves_horizontal_camera_orientation(self):
        s = self.session()
        s._pose = {"position": [2.,0.,3.], "rotation": [1.,0.,0.,0.]}
        s.plan = [{"label": "desk"}]
        rotations = []
        sim = SimpleNamespace(
            set_agent_state=lambda position, rotation: rotations.append(list(rotation)),
            get_observations_at=lambda **kwargs: rotations.append(kwargs["rotation"]),
        )
        s.env = SimpleNamespace(sim=sim, reset=lambda: None)
        s.pose = lambda: s._pose
        s._continue_after_episode_end()
        self.assertEqual(len(rotations), 2)
        # The simulator's identity orientation is xyzw=[0,0,0,1], not [1,0,0,0].
        self.assertEqual(rotations, [[0.,0.,0.,1.], [0.,0.,0.,1.]])
        self.assertEqual(s.events[-1]["kind"], "round_advanced")


if __name__ == "__main__":
    unittest.main()
