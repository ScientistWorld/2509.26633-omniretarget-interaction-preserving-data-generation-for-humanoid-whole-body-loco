Split: test. Rank delta vs `omniretarget` (paired 95% CI).

| method | status | CPU s | cohort6 | 95% CI | nearref | 95% CI |
|---|---|---:|---:|---|---:|---|
| sc_select_best | ok  | 450.1 | +0.225 | [+0.161, +0.297] | +6.306 | [+4.713, +8.012] |
| var_keypoint_contact_w100 | ok  | 114.3 | +0.170 | [+0.104, +0.249] | +1.482 | [-1.031, +3.947] |
| var_contact_aware | ok  | 136.4 | +0.153 | [+0.088, +0.229] | +1.171 | [-0.350, +2.769] |
| var_keypoint_contact | ok  | 111.4 | +0.054 | [-0.036, +0.145] | -1.381 | [-3.987, +1.092] |
| var_keypoint_contact_it11 | ok  | 91.1 | +0.002 | [-0.071, +0.081] | -1.071 | [-3.582, +1.467] |
| omniretarget | ok  | 126.7 | +0.000 |  | +0.000 |  |
| var_no_mesh | ok  | 103.5 | -0.566 | [-0.729, -0.413] | -4.782 | [-7.675, -2.025] |
| sc_constant_pose | ok  | 1.4 | -6.255 | [-6.505, -5.994] | -566.634 | [-603.255, -529.850] |
| keypoint_ik | ok  | 62.2 | -8.311 | [-9.116, -7.558] | -1101.969 | [-1205.235, -1005.485] |
| sc_disconnected_keypoints | ok  | 61.2 | -8.311 | [-9.116, -7.558] | -1101.969 | [-1205.235, -1005.485] |
| sc_score_forgery | ok  | 59.9 | -8.311 | [-9.116, -7.558] | -1101.969 | [-1205.235, -1005.485] |
| sc_empty | failed test_obj_00: missing output | 1.4 | -120.714 |  | -30922.977 |  |
| sc_impossible_q | failed test_obj_00: joint limit violated | 1.4 | -120.714 |  | -30922.977 |  |
| sc_joint_violation | failed test_obj_00: joint limit violated | 59.9 | -120.714 |  | -30922.977 |  |
| sc_malformed | failed test_obj_00: q has shape (0,), expected (85, 11) | 1.3 | -120.714 |  | -30922.977 |  |
| sc_nan | failed test_obj_00: non-finite configuration | 1.2 | -120.714 |  | -30922.977 |  |
| sc_omit_failed_experiment | failed test_ter_00: missing output | 38.8 | -120.714 |  | -30922.977 |  |
| sc_partial_missing | failed test_ter_00: missing output | 73.1 | -120.714 |  | -30922.977 |  |
| sc_velocity_violation | failed test_obj_00: velocity limit violated | 59.9 | -120.714 |  | -30922.977 |  |
| sc_zero_q | failed test_obj_00: joint limit violated | 1.4 | -120.714 |  | -30922.977 |  |

Per-experiment metric means:

**object_interaction_kinematics**

| method | penetration_duration | max_penetration_cm | foot_skating_duration | max_foot_skating_cm_s | contact_preservation | spurious_contact_duration | pelvis_path_error_cm | motion_velocity_error_cm_s |
|---|---|---|---|---|---|---|---|---|
| keypoint_ik | 0.4661 | 1.4859 | 0.1476 | 6.0697 | 0.6271 | 0.0000 | 0.2230 | 0.6334 |
| omniretarget | 0.0000 | 0.0038 | 0.0013 | 0.5219 | 0.9926 | 0.0005 | 0.7058 | 2.9882 |
| sc_constant_pose | 0.0392 | 0.3873 | 0.0000 | 0.0000 | 0.4129 | 1.0000 | 23.1181 | 21.5914 |
| sc_disconnected_keypoints | 0.4661 | 1.4859 | 0.1476 | 6.0697 | 0.6271 | 0.0000 | 0.2230 | 0.6334 |
| sc_score_forgery | 0.4661 | 1.4859 | 0.1476 | 6.0697 | 0.6271 | 0.0000 | 0.2230 | 0.6334 |
| sc_select_best | 0.0000 | 0.0051 | 0.0001 | 0.2549 | 1.0000 | 0.0005 | 0.4916 | 2.9473 |
| var_contact_aware | 0.0000 | 0.0042 | 0.0019 | 0.5293 | 1.0000 | 0.0005 | 0.5842 | 2.8440 |
| var_keypoint_contact | 0.0000 | 0.0061 | 0.0023 | 0.5743 | 1.0000 | 0.0043 | 0.3732 | 3.2517 |
| var_keypoint_contact_it11 | 0.0000 | 0.0067 | 0.0024 | 0.5188 | 1.0000 | 0.0043 | 0.3618 | 3.2510 |
| var_keypoint_contact_w100 | 0.0000 | 0.0064 | 0.0006 | 0.4464 | 1.0000 | 0.0043 | 0.3643 | 3.3257 |
| var_no_mesh | 0.0000 | 0.0067 | 0.0013 | 0.4528 | 0.9561 | 0.0039 | 0.4950 | 3.4286 |

**terrain_interaction_kinematics**

| method | penetration_duration | max_penetration_cm | foot_skating_duration | max_foot_skating_cm_s | contact_preservation | spurious_contact_duration | pelvis_path_error_cm | motion_velocity_error_cm_s |
|---|---|---|---|---|---|---|---|---|
| keypoint_ik | 0.0984 | 0.8209 | 0.1713 | 14.4116 | 0.9994 | 0.0001 | 0.2589 | 1.8749 |
| omniretarget | 0.0000 | 0.0000 | 0.0016 | 0.5626 | 0.9893 | 0.0004 | 0.6856 | 6.7785 |
| sc_constant_pose | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8145 | 0.1975 | 29.0071 | 28.5071 |
| sc_disconnected_keypoints | 0.0984 | 0.8209 | 0.1713 | 14.4116 | 0.9994 | 0.0001 | 0.2589 | 1.8749 |
| sc_score_forgery | 0.0984 | 0.8209 | 0.1713 | 14.4116 | 0.9994 | 0.0001 | 0.2589 | 1.8749 |
| sc_select_best | 0.0000 | 0.0001 | 0.0000 | 0.2412 | 1.0000 | 0.0014 | 0.4879 | 6.2430 |
| var_contact_aware | 0.0000 | 0.0000 | 0.0004 | 0.4795 | 1.0000 | 0.0005 | 0.5928 | 7.0090 |
| var_keypoint_contact | 0.0000 | 0.0001 | 0.0010 | 0.4898 | 1.0000 | 0.0018 | 0.4955 | 6.1644 |
| var_keypoint_contact_it11 | 0.0000 | 0.0000 | 0.0016 | 0.5081 | 1.0000 | 0.0018 | 0.4929 | 6.1715 |
| var_keypoint_contact_w100 | 0.0000 | 0.0000 | 0.0000 | 0.2630 | 1.0000 | 0.0017 | 0.4603 | 6.3324 |
| var_no_mesh | 0.0000 | 0.0007 | 0.0017 | 0.6114 | 0.9941 | 0.0018 | 0.5980 | 6.0606 |

**augmented_interaction_kinematics**

| method | penetration_duration | max_penetration_cm | foot_skating_duration | max_foot_skating_cm_s | contact_preservation | spurious_contact_duration | pelvis_path_error_cm | motion_velocity_error_cm_s |
|---|---|---|---|---|---|---|---|---|
| keypoint_ik | 0.5787 | 5.6802 | 0.1522 | 11.7854 | 0.5208 | 0.0030 | 1.2703 | 1.4342 |
| omniretarget | 0.0171 | 0.1906 | 0.0010 | 0.5702 | 0.9668 | 0.0095 | 1.9816 | 7.0985 |
| sc_constant_pose | 0.0488 | 0.6577 | 0.0000 | 0.0000 | 0.6126 | 0.6008 | 26.6591 | 26.0344 |
| sc_disconnected_keypoints | 0.5787 | 5.6802 | 0.1522 | 11.7854 | 0.5208 | 0.0030 | 1.2703 | 1.4342 |
| sc_score_forgery | 0.5787 | 5.6802 | 0.1522 | 11.7854 | 0.5208 | 0.0030 | 1.2703 | 1.4342 |
| sc_select_best | 0.0000 | 0.0062 | 0.0002 | 0.3150 | 1.0000 | 0.0027 | 1.5614 | 7.3529 |
| var_contact_aware | 0.0000 | 0.0192 | 0.0018 | 0.6669 | 1.0000 | 0.0102 | 1.9139 | 7.3552 |
| var_keypoint_contact | 0.0058 | 0.2716 | 0.0018 | 1.5500 | 0.9726 | 0.0118 | 1.3315 | 7.1380 |
| var_keypoint_contact_it11 | 0.0085 | 0.2782 | 0.0012 | 0.4437 | 0.9191 | 0.0118 | 1.4183 | 6.7884 |
| var_keypoint_contact_w100 | 0.0006 | 0.0547 | 0.0010 | 0.6648 | 1.0000 | 0.0118 | 1.3331 | 7.6666 |
| var_no_mesh | 0.2036 | 0.6378 | 0.0019 | 1.0064 | 0.8271 | 0.0118 | 1.4189 | 6.5235 |

