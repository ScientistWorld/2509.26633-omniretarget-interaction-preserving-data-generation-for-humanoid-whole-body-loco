| method | status | rank delta vs reference | paired 95% CI |
|---|---|---:|---|
| var_keypoint_contact | ok  | +0.979 | [+0.592, +1.402] |
| var_contact_aware | ok  | +0.955 | [+0.558, +1.398] |
| omniretarget | ok  | +0.000 |  |
| var_no_mesh | ok  | -0.505 | [-1.030, +0.053] |
| sc_constant_pose | ok  | -3.915 | [-5.037, -2.944] |
| keypoint_ik | ok  | -7.779 | [-9.169, -6.377] |
| sc_disconnected_keypoints | ok  | -7.779 | [-9.232, -6.439] |
| sc_score_forgery | ok  | -7.779 | [-9.210, -6.436] |
| sc_empty | failed test_obj_00: missing output | -91.972 |  |
| sc_impossible_q | failed test_obj_00: joint limit violated | -91.972 |  |
| sc_joint_violation | failed test_obj_00: joint limit violated | -91.972 |  |
| sc_malformed | failed test_obj_00: q has shape (0,), expected (82, 11) | -91.972 |  |
| sc_nan | failed test_obj_00: non-finite configuration | -91.972 |  |
| sc_omit_failed_experiment | failed test_ter_00: missing output | -91.972 |  |
| sc_partial_missing | failed test_ter_00: missing output | -91.972 |  |
| sc_velocity_violation | failed test_obj_00: velocity limit violated | -91.972 |  |
| sc_zero_q | failed test_obj_00: joint limit violated | -91.972 |  |
