/**
 * 静态快照（自动生成，请勿手改）。
 *
 * 生成命令：`node scripts/refresh-cache.mjs`
 * 数据来源：导航服务器 `agh_nav_probe.py board`
 * 生成时间：2026-10-11T10:55:10.442Z
 */

export const BASELINE = {
  "generated_at_iso": "2026-10-11T18:55:08+0800",
  "payload": {
    "schema": "agh-nav-probe/1",
    "command": "board",
    "generated_at_iso": "2026-10-11T18:55:08+0800",
    "evaluation_protocol": {
      "manifest": "val_unseen_multiscene（主清单 4 场景×13 回合）+ val_unseen_multiscene_b（新场景 4×13）",
      "episodes_per_config": 104,
      "seeds": "20260826",
      "metric": "Habitat VLN Success / SPL / NE / 碰撞 / 步数 / System-2 调用",
      "note": "4 个调参场景 + 4 个模型没见过的场景；每格 13 回合，SE≈14pt"
    },
    "recent_runs": [
      {
        "label": "r1col4b",
        "dir": "/srv/agh-demo/InternNav/logs/habitat/route1_r1col4b_seed20260826",
        "mode": "recovery_only",
        "episodes": 52,
        "expected_episodes": 52,
        "n_scenes": 4,
        "pooled_success": 0.6731,
        "pooled_spl": 0.5873,
        "worst_scene_success": 0.6154,
        "collision_per_ep": 8.19,
        "system2_per_ep": 23.5,
        "mtime_iso": "2026-10-11T18:31:16+0800"
      },
      {
        "label": "r1col4",
        "dir": "/srv/agh-demo/InternNav/logs/habitat/route1_r1col4_seed20260826",
        "mode": "recovery_only",
        "episodes": 52,
        "expected_episodes": 52,
        "n_scenes": 4,
        "pooled_success": 0.6346,
        "pooled_spl": 0.5609,
        "worst_scene_success": 0.4615,
        "collision_per_ep": 20.31,
        "system2_per_ep": 20.46,
        "mtime_iso": "2026-10-11T17:22:26+0800"
      },
      {
        "label": "r1ro4c",
        "dir": "/srv/agh-demo/InternNav/logs/habitat/route1_r1ro4c_seed20260826",
        "mode": "recovery_only",
        "episodes": 52,
        "expected_episodes": 52,
        "n_scenes": 4,
        "pooled_success": 0.6538,
        "pooled_spl": 0.5825,
        "worst_scene_success": 0.5385,
        "collision_per_ep": 11.63,
        "system2_per_ep": 23.4,
        "mtime_iso": "2026-10-11T16:06:46+0800"
      },
      {
        "label": "r1orig4b",
        "dir": "/srv/agh-demo/InternNav/logs/habitat/route1_r1orig4b_seed20260826",
        "mode": "original",
        "episodes": 52,
        "expected_episodes": 52,
        "n_scenes": 4,
        "pooled_success": 0.6538,
        "pooled_spl": 0.5894,
        "worst_scene_success": 0.4615,
        "collision_per_ep": 14.42,
        "system2_per_ep": 28.38,
        "mtime_iso": "2026-10-11T14:31:29+0800"
      },
      {
        "label": "r1multi4b",
        "dir": "/srv/agh-demo/InternNav/logs/habitat/route1_r1multi4b_seed20260826",
        "mode": "drift_recovery_periodic",
        "episodes": 52,
        "expected_episodes": 52,
        "n_scenes": 4,
        "pooled_success": 0.6154,
        "pooled_spl": 0.5376,
        "worst_scene_success": 0.5385,
        "collision_per_ep": 0.85,
        "system2_per_ep": 23.44,
        "mtime_iso": "2026-10-11T13:09:58+0800"
      },
      {
        "label": "r1grd4b",
        "dir": "/srv/agh-demo/InternNav/logs/habitat/route1_r1grd4b_seed20260826",
        "mode": "recovery_only",
        "episodes": 52,
        "expected_episodes": 52,
        "n_scenes": 4,
        "pooled_success": 0.6731,
        "pooled_spl": 0.6053,
        "worst_scene_success": 0.6154,
        "collision_per_ep": 8.33,
        "system2_per_ep": 24.62,
        "mtime_iso": "2026-10-11T01:58:51+0800"
      },
      {
        "label": "r1ro4b",
        "dir": "/srv/agh-demo/InternNav/logs/habitat/route1_r1ro4b_seed20260826",
        "mode": "recovery_only",
        "episodes": 52,
        "expected_episodes": 52,
        "n_scenes": 4,
        "pooled_success": 0.5577,
        "pooled_spl": 0.5061,
        "worst_scene_success": 0.3846,
        "collision_per_ep": 10.06,
        "system2_per_ep": 25.29,
        "mtime_iso": "2026-10-11T00:50:23+0800"
      },
      {
        "label": "r1grd4",
        "dir": "/srv/agh-demo/InternNav/logs/habitat/route1_r1grd4_seed20260826",
        "mode": "recovery_only",
        "episodes": 52,
        "expected_episodes": 52,
        "n_scenes": 4,
        "pooled_success": 0.75,
        "pooled_spl": 0.667,
        "worst_scene_success": 0.4615,
        "collision_per_ep": 8.04,
        "system2_per_ep": 19.46,
        "mtime_iso": "2026-10-10T21:18:23+0800"
      }
    ],
    "scenes": [
      "2azQ1b91cZZ",
      "8194nk5LbLH",
      "EU6Fwq7SyZv",
      "QUCTc6BB5sX",
      "TbHJrupSAjP",
      "X7HyMhZNoso",
      "Z6MFQCViBuw",
      "zsNo4HB9uLZ"
    ],
    "configs": {
      "R2A": {
        "source_file": "pool8_r1multi4.json",
        "pooled_success": 0.6058,
        "pooled_spl": 0.5421,
        "episodes_total": 104,
        "scenes_total": 8,
        "spread": {
          "n_scenes": 8,
          "min": 0.1538,
          "max": 0.8462,
          "std": 0.2114,
          "range": 0.6923
        },
        "per_scene": {
          "2azQ1b91cZZ": {
            "success": 0.6923,
            "episodes": 13,
            "collision": 0.62,
            "steps": 75.3,
            "system2_calls": 17,
            "terminations": {
              "success": 9,
              "budget_exhausted_no_goal": 4
            }
          },
          "QUCTc6BB5sX": {
            "success": 0.1538,
            "episodes": 13,
            "collision": 3,
            "steps": 83.3,
            "system2_calls": 16.46,
            "terminations": {
              "budget_exhausted_no_goal": 6,
              "success": 2,
              "watchdog_stop": 5
            }
          },
          "TbHJrupSAjP": {
            "success": 0.6923,
            "episodes": 13,
            "collision": 0,
            "steps": 75.5,
            "system2_calls": 19.38,
            "terminations": {
              "budget_exhausted_no_goal": 4,
              "success": 9
            }
          },
          "zsNo4HB9uLZ": {
            "success": 0.8462,
            "episodes": 13,
            "collision": 0,
            "steps": 84.5,
            "system2_calls": 18.15,
            "terminations": {
              "success": 11,
              "budget_exhausted_no_goal": 2
            }
          },
          "8194nk5LbLH": {
            "success": 0.5385,
            "episodes": 13,
            "collision": 1.46,
            "steps": 165.1,
            "system2_calls": 40.54,
            "terminations": {
              "success": 7,
              "watchdog_stop": 1,
              "budget_exhausted_no_goal": 5
            }
          },
          "EU6Fwq7SyZv": {
            "success": 0.5385,
            "episodes": 13,
            "collision": 1.23,
            "steps": 102.8,
            "system2_calls": 23.69,
            "terminations": {
              "success": 7,
              "budget_exhausted_no_goal": 6
            }
          },
          "X7HyMhZNoso": {
            "success": 0.7692,
            "episodes": 13,
            "collision": 0.54,
            "steps": 76.9,
            "system2_calls": 18.69,
            "terminations": {
              "success": 10,
              "budget_exhausted_no_goal": 3
            }
          },
          "Z6MFQCViBuw": {
            "success": 0.6154,
            "episodes": 13,
            "collision": 0.15,
            "steps": 57.5,
            "system2_calls": 10.85,
            "terminations": {
              "budget_exhausted_no_goal": 5,
              "success": 8
            }
          }
        },
        "source_runs": [
          "route1_r1multi4_seed20260826",
          "route1_r1multi4b_seed20260826"
        ]
      },
      "original": {
        "source_file": "pool8_r1orig4.json",
        "pooled_success": 0.6346,
        "pooled_spl": 0.5853,
        "episodes_total": 104,
        "scenes_total": 8,
        "spread": {
          "n_scenes": 8,
          "min": 0.2308,
          "max": 0.8462,
          "std": 0.2046,
          "range": 0.6154
        },
        "per_scene": {
          "2azQ1b91cZZ": {
            "success": 0.6923,
            "episodes": 13,
            "collision": 0.31,
            "steps": 75.8,
            "system2_calls": 20.08,
            "terminations": {
              "budget_exhausted_no_goal": 4,
              "success": 9
            }
          },
          "QUCTc6BB5sX": {
            "success": 0.2308,
            "episodes": 13,
            "collision": 76.46,
            "steps": 216,
            "system2_calls": 51.23,
            "terminations": {
              "success": 3,
              "budget_exhausted_no_goal": 10
            }
          },
          "TbHJrupSAjP": {
            "success": 0.6923,
            "episodes": 13,
            "collision": 1.62,
            "steps": 66.5,
            "system2_calls": 17.38,
            "terminations": {
              "budget_exhausted_no_goal": 4,
              "success": 9
            }
          },
          "zsNo4HB9uLZ": {
            "success": 0.8462,
            "episodes": 13,
            "collision": 0,
            "steps": 87.1,
            "system2_calls": 22.15,
            "terminations": {
              "success": 11,
              "budget_exhausted_no_goal": 2
            }
          },
          "8194nk5LbLH": {
            "success": 0.6923,
            "episodes": 13,
            "collision": 25.31,
            "steps": 185.8,
            "system2_calls": 46.23,
            "terminations": {
              "success": 9,
              "budget_exhausted_no_goal": 4
            }
          },
          "EU6Fwq7SyZv": {
            "success": 0.4615,
            "episodes": 13,
            "collision": 32.08,
            "steps": 143.5,
            "system2_calls": 35,
            "terminations": {
              "success": 6,
              "budget_exhausted_no_goal": 7
            }
          },
          "X7HyMhZNoso": {
            "success": 0.8462,
            "episodes": 13,
            "collision": 0.31,
            "steps": 61.5,
            "system2_calls": 17.23,
            "terminations": {
              "success": 11,
              "budget_exhausted_no_goal": 2
            }
          },
          "Z6MFQCViBuw": {
            "success": 0.6154,
            "episodes": 13,
            "collision": 0,
            "steps": 60.5,
            "system2_calls": 15.08,
            "terminations": {
              "budget_exhausted_no_goal": 5,
              "success": 8
            }
          }
        },
        "source_runs": [
          "route1_r1orig4_seed20260826",
          "route1_r1orig4b_seed20260826"
        ]
      },
      "recovery_only": {
        "source_file": "pool8_r1ro4c.json",
        "pooled_success": 0.6635,
        "pooled_spl": 0.5999,
        "episodes_total": 104,
        "scenes_total": 8,
        "spread": {
          "n_scenes": 8,
          "min": 0.5385,
          "max": 0.7692,
          "std": 0.0914,
          "range": 0.2308
        },
        "per_scene": {
          "2azQ1b91cZZ": {
            "success": 0.6923,
            "episodes": 13,
            "collision": 1.23,
            "steps": 92.5,
            "system2_calls": 17.54,
            "terminations": {
              "success": 9,
              "budget_exhausted_no_goal": 4
            }
          },
          "QUCTc6BB5sX": {
            "success": 0.5385,
            "episodes": 13,
            "collision": 54.69,
            "steps": 165.2,
            "system2_calls": 15.85,
            "terminations": {
              "budget_exhausted_no_goal": 6,
              "success": 7
            }
          },
          "TbHJrupSAjP": {
            "success": 0.6923,
            "episodes": 13,
            "collision": 21.31,
            "steps": 117.3,
            "system2_calls": 19.23,
            "terminations": {
              "budget_exhausted_no_goal": 4,
              "success": 9
            }
          },
          "zsNo4HB9uLZ": {
            "success": 0.7692,
            "episodes": 13,
            "collision": 0,
            "steps": 93.4,
            "system2_calls": 18.08,
            "terminations": {
              "success": 10,
              "budget_exhausted_no_goal": 3
            }
          },
          "8194nk5LbLH": {
            "success": 0.6154,
            "episodes": 13,
            "collision": 38.46,
            "steps": 252.3,
            "system2_calls": 50.92,
            "terminations": {
              "success": 8,
              "budget_exhausted_no_goal": 5
            }
          },
          "EU6Fwq7SyZv": {
            "success": 0.5385,
            "episodes": 13,
            "collision": 1.15,
            "steps": 72.4,
            "system2_calls": 14.77,
            "terminations": {
              "success": 7,
              "budget_exhausted_no_goal": 6
            }
          },
          "X7HyMhZNoso": {
            "success": 0.7692,
            "episodes": 13,
            "collision": 6.92,
            "steps": 85.5,
            "system2_calls": 18.15,
            "terminations": {
              "success": 10,
              "budget_exhausted_no_goal": 3
            }
          },
          "Z6MFQCViBuw": {
            "success": 0.6923,
            "episodes": 13,
            "collision": 0,
            "steps": 54.8,
            "system2_calls": 9.77,
            "terminations": {
              "budget_exhausted_no_goal": 4,
              "success": 9
            }
          }
        },
        "source_runs": [
          "route1_r1ro4_seed20260826",
          "route1_r1ro4c_seed20260826"
        ]
      },
      "guarded": {
        "source_file": "pool8_r1grd4.json",
        "pooled_success": 0.7115,
        "pooled_spl": 0.6361,
        "episodes_total": 104,
        "scenes_total": 8,
        "spread": {
          "n_scenes": 8,
          "min": 0.4615,
          "max": 1,
          "std": 0.178,
          "range": 0.5385
        },
        "per_scene": {
          "2azQ1b91cZZ": {
            "success": 0.9231,
            "episodes": 13,
            "collision": 2.31,
            "steps": 74.4,
            "system2_calls": 14.92,
            "terminations": {
              "success": 12,
              "budget_exhausted_no_goal": 1
            }
          },
          "QUCTc6BB5sX": {
            "success": 0.4615,
            "episodes": 13,
            "collision": 28.54,
            "steps": 141.5,
            "system2_calls": 27.69,
            "terminations": {
              "budget_exhausted_no_goal": 7,
              "success": 6
            }
          },
          "TbHJrupSAjP": {
            "success": 0.6154,
            "episodes": 13,
            "collision": 1.31,
            "steps": 60.2,
            "system2_calls": 15.46,
            "terminations": {
              "budget_exhausted_no_goal": 5,
              "success": 8
            }
          },
          "zsNo4HB9uLZ": {
            "success": 1,
            "episodes": 13,
            "collision": 0,
            "steps": 93.5,
            "system2_calls": 19.77,
            "terminations": {
              "success": 13
            }
          },
          "8194nk5LbLH": {
            "success": 0.6154,
            "episodes": 13,
            "collision": 30.38,
            "steps": 236.3,
            "system2_calls": 50.08,
            "terminations": {
              "success": 8,
              "budget_exhausted_no_goal": 5
            }
          },
          "EU6Fwq7SyZv": {
            "success": 0.6154,
            "episodes": 13,
            "collision": 1,
            "steps": 96.2,
            "system2_calls": 22.31,
            "terminations": {
              "success": 8,
              "budget_exhausted_no_goal": 5
            }
          },
          "X7HyMhZNoso": {
            "success": 0.7692,
            "episodes": 13,
            "collision": 1.92,
            "steps": 57.5,
            "system2_calls": 15.77,
            "terminations": {
              "success": 10,
              "budget_exhausted_no_goal": 3
            }
          },
          "Z6MFQCViBuw": {
            "success": 0.6923,
            "episodes": 13,
            "collision": 0,
            "steps": 54.1,
            "system2_calls": 10.31,
            "terminations": {
              "budget_exhausted_no_goal": 4,
              "success": 9
            }
          }
        },
        "source_runs": [
          "route1_r1grd4_seed20260826",
          "route1_r1grd4b_seed20260826"
        ]
      },
      "collision_only": {
        "source_file": "pool8_r1col4.json",
        "pooled_success": 0.6538,
        "pooled_spl": 0.5741,
        "episodes_total": 104,
        "scenes_total": 8,
        "spread": {
          "n_scenes": 8,
          "min": 0.4615,
          "max": 0.8462,
          "std": 0.1163,
          "range": 0.3846
        },
        "per_scene": {
          "2azQ1b91cZZ": {
            "success": 0.6923,
            "episodes": 13,
            "collision": 1.23,
            "steps": 92.5,
            "system2_calls": 17.54,
            "terminations": {
              "success": 9,
              "budget_exhausted_no_goal": 4
            }
          },
          "QUCTc6BB5sX": {
            "success": 0.4615,
            "episodes": 13,
            "collision": 78.69,
            "steps": 191.7,
            "system2_calls": 21.15,
            "terminations": {
              "budget_exhausted_no_goal": 7,
              "success": 6
            }
          },
          "TbHJrupSAjP": {
            "success": 0.5385,
            "episodes": 13,
            "collision": 1.15,
            "steps": 96.8,
            "system2_calls": 21.46,
            "terminations": {
              "budget_exhausted_no_goal": 6,
              "success": 7
            }
          },
          "zsNo4HB9uLZ": {
            "success": 0.8462,
            "episodes": 13,
            "collision": 0.15,
            "steps": 103.5,
            "system2_calls": 21.69,
            "terminations": {
              "success": 11,
              "budget_exhausted_no_goal": 2
            }
          },
          "8194nk5LbLH": {
            "success": 0.6154,
            "episodes": 13,
            "collision": 29.92,
            "steps": 246.1,
            "system2_calls": 51.54,
            "terminations": {
              "success": 8,
              "budget_exhausted_no_goal": 5
            }
          },
          "EU6Fwq7SyZv": {
            "success": 0.6923,
            "episodes": 13,
            "collision": 1.23,
            "steps": 72.3,
            "system2_calls": 15.46,
            "terminations": {
              "success": 9,
              "budget_exhausted_no_goal": 4
            }
          },
          "X7HyMhZNoso": {
            "success": 0.6923,
            "episodes": 13,
            "collision": 1.46,
            "steps": 77.6,
            "system2_calls": 18,
            "terminations": {
              "success": 9,
              "budget_exhausted_no_goal": 4
            }
          },
          "Z6MFQCViBuw": {
            "success": 0.6923,
            "episodes": 13,
            "collision": 0.15,
            "steps": 52.9,
            "system2_calls": 9,
            "terminations": {
              "budget_exhausted_no_goal": 4,
              "success": 9
            }
          }
        },
        "source_runs": [
          "route1_r1col4_seed20260826",
          "route1_r1col4b_seed20260826"
        ]
      }
    },
    "final_table_txt": "================================================================\n 路线① 导航侧 最终对照表   生成于 2026-10-11T18:33:31+08:00\n 口径：8 场景 × 13 回合 = 104 回合/配置（4 个是调参场景，4 个是模型没见过的场景）\n================================================================\n  scene            ep         R2A    original recovery_only    guarded collision_only\n  2azQ1b91cZZ      13      0.6923      0.6923        0.6923     0.9231         0.6923\n  8194nk5LbLH      13      0.5385      0.6923        0.6154     0.6154         0.6154\n  EU6Fwq7SyZv      13      0.5385      0.4615        0.5385     0.6154         0.6923\n  QUCTc6BB5sX      13      0.1538      0.2308        0.5385     0.4615         0.4615\n  TbHJrupSAjP      13      0.6923      0.6923        0.6923     0.6154         0.5385\n  X7HyMhZNoso      13      0.7692      0.8462        0.7692     0.7692         0.6923\n  Z6MFQCViBuw      13      0.6154      0.6154        0.6923     0.6923         0.6923\n  zsNo4HB9uLZ      13      0.8462      0.8462        0.7692     1.0000         0.8462\n\n  池化 success      104      0.6058      0.6346        0.6635     0.7115         0.6538\n  池化 SPL          104      0.5421      0.5853        0.5999     0.6361         0.5741\n  最差场景              -      0.1538      0.2308        0.5385     0.4615         0.4615\n  最好场景              -      0.8462      0.8462        0.7692     1.0000         0.8462\n  场景间 std           -      0.2114      0.2046        0.0914     0.1780         0.1163\n  场景间 range         -      0.6923      0.6154        0.2308     0.5385         0.3846\n\n  代价（8 场景按回合数加权平均）:\n    R2A              coll=  0.88  steps=  90.1  S2= 20.6  replans=  5.5  NE= 3.88\n    original         coll= 17.01  steps= 112.1  S2= 28.0  replans=  0.0  NE= 4.13\n    recovery_only    coll= 15.47  steps= 116.7  S2= 20.5  replans=  4.9  NE= 3.86\n    guarded          coll=  8.18  steps= 101.7  S2= 22.0  replans=  5.9  NE= 3.50\n    collision_only   coll= 14.25  steps= 116.7  S2= 22.0  replans=  5.6  NE= 3.84\n\n  只看新 4 场景（模型没见过的场景，每臂 52 回合）:\n    R2A              success=0.6154\n    original         success=0.6538\n    recovery_only    success=0.6538\n    guarded          success=0.6731\n    collision_only   success=0.6731\n\n--- 仅碰撞通道：主清单 run 的失败模式 ---\n== route1_r1col4_seed20260826 ==\n  mode=recovery_only  safety={\"max_consecutive_collision_actions\": 0, \"collision_window_actions\": 0, \"max_collisions_in_window\": 0}  horizon(最大 steps)=501  场景数=4\n  池化：52 回合 success=0.6346 spl=0.5609 ne=3.891 coll=20.31 steps=121.1 replans=6.4\n  终止方式分布: {\"success\": 33, \"budget_exhausted_no_goal\": 19}\n\n  scene           ep success   coll  steps   repl     S2  wd_stop 终止方式\n  2azQ1b91cZZ     13  0.6923   1.23   92.5    5.3   17.5        0 {\"success\": 9, \"budget_exhausted_no_goal\": 4}\n  QUCTc6BB5sX     13  0.4615  78.69  191.7    8.3   21.2        0 {\"budget_exhausted_no_goal\": 7, \"success\": 6}\n  TbHJrupSAjP     13  0.5385   1.15   96.8    5.5   21.5        0 {\"budget_exhausted_no_goal\": 6, \"success\": 7}\n  zsNo4HB9uLZ     13  0.8462   0.15  103.5    6.6   21.7        0 {\"success\": 11, \"budget_exhausted_no_goal\": 2}\n\n  场景间 success：min=0.4615 max=0.8462 std=0.1706 range=0.3846（n=4 场景）\n\n  场景内：成功 vs 失败 的均值差（collision / replans / no_action / steps）\n    2azQ1b91cZZ    succ n= 9 fail n= 4 | coll 1.67 vs 0.25 | replans 3.6 vs 9.2 | no_action 0.9 vs 0.5 | steps 65.6 vs 153.2\n        相关(success, ·): collision_count=0.164, replans=-0.829, no_action_count=0.258, system2_calls=-0.837, steps=-0.842\n    QUCTc6BB5sX    succ n= 6 fail n= 7 | coll 61.67 vs 93.29 | replans 8.8 vs 7.9 | no_action 0.7 vs 0.3 | steps 151.0 vs 226.6\n        相关(success, ·): collision_count=-0.140, replans=0.149, no_action_count=0.381, system2_calls=-0.081, steps=-0.265\n    TbHJrupSAjP    succ n= 7 fail n= 6 | coll 0.00 vs 2.50 | replans 3.9 vs 7.5 | no_action 0.1 vs 1.2 | steps 53.4 vs 147.5\n        相关(success, ·): collision_count=-0.335, replans=-0.451, no_action_count=-0.474, system2_calls=-0.434, steps=-0.394\n    zsNo4HB9uLZ    succ n=11 fail n= 2 | coll 0.18 vs 0.00 | replans 6.8 vs 5.5 | no_action 0.9 vs 0.5 | steps 109.2 vs 72.0\n        相关(success, ·): collision_count=0.123, replans=0.130, no_action_count=0.114, system2_calls=0.243, steps=0.268\n\n  失败回合明细（含终止证据）\n    2azQ1b91cZZ    ep43   budget_exhausted_no_goal steps=125 last_action=5    stops= 1 tail=[2, 2, 2, 2, 5] coll=1 wd=0 replans=9 no_action=0 ne=5.74\n    2azQ1b91cZZ    ep44   budget_exhausted_no_goal steps= 96 last_action=5    stops= 1 tail=[2, 2, 2, 2, 5] coll=0 wd=0 replans=7 no_action=1 ne=15.45\n    2azQ1b91cZZ    ep61   budget_exhausted_no_goal steps=186 last_action=5    stops= 1 tail=[2, 2, 2, 2, 5] coll=0 wd=0 replans=7 no_action=0 ne=11.19\n    2azQ1b91cZZ    ep70   budget_exhausted_no_goal steps=206 last_action=5    stops= 1 tail=[2, 2, 2, 2, 5] coll=0 wd=0 replans=14 no_action=1 ne=12.09\n\n--- 仅碰撞通道：通道触发统计（两个清单） ---\n  route1_r1col4      {'recovery_only_observe': 1407, 'recovery': 35, 'local_horizon_replan': 260, 'collision_density_replan': 40}\n  route1_r1col4b     {'recovery_only_observe': 926, 'local_horizon_replan': 208, 'recovery': 25, 'collision_density_replan': 12}\n\n--- 队列日志尾部 ---\n[18:31:16:397081]:[Sim] Simulator.cpp(69)::~Simulator : Deconstructing Simulator\n[18:31:16.397635] [overlay_v4] final stats: {'forced_replans': 12, 'forced_reasons': {'collision_density_replan': 12}}\n[r1col4b] EXIT=0 2026-10-11T18:31:17+08:00\nALL_COL_DONE 2026-10-11T18:31:17+08:00\n"
  }
}
