| method | status | rank delta vs reference | paired 95% CI |
|---|---|---:|---|
| var_keypoint_contact | ok  | +1.659 | [+1.258, +2.060] |
| var_contact_aware | ok  | +1.472 | [+1.074, +1.898] |
| omniretarget | ok  | +0.000 |  |
| var_no_mesh | ok  | -0.138 | [-0.557, +0.283] |
| sc_constant_pose | ok  | -3.336 | [-3.952, -2.724] |
| sc_disconnected_keypoints | ok  | -7.385 | [-8.225, -6.610] |
| sc_score_forgery | ok  | -7.385 | [-8.208, -6.642] |
| keypoint_ik | ok  | -7.385 | [-8.181, -6.567] |
| sc_empty | failed test_obj_00: missing output | -84.267 |  |
| sc_impossible_q | failed test_obj_00: joint limit violated | -84.267 |  |
| sc_joint_violation | failed test_obj_00: joint limit violated | -84.267 |  |
| sc_malformed | failed test_obj_00: q has shape (0,), expected (82, 11) | -84.267 |  |
| sc_nan | failed test_obj_00: non-finite configuration | -84.267 |  |
| sc_omit_failed_experiment | failed test_ter_00: missing output | -84.267 |  |
| sc_partial_missing | failed test_ter_00: missing output | -84.267 |  |
| sc_velocity_violation | failed test_obj_00: velocity limit violated | -84.267 |  |
| sc_zero_q | failed test_obj_00: joint limit violated | -84.267 |  |
