Split: train. Rank delta vs `omniretarget` (paired 95% CI).

| method | status | CPU s | cohort6 | 95% CI | nearref | 95% CI |
|---|---|---:|---:|---|---:|---|
| sc_select_best | ok  | 151.5 | +0.239 | [+0.145, +0.350] | +4.808 | [+2.866, +6.758] |
| var_contact_aware | ok  | 41.5 | +0.180 | [+0.085, +0.290] | +2.286 | [+0.446, +4.297] |
| var_keypoint_contact_w100 | ok  | 37.1 | +0.174 | [+0.067, +0.292] | +3.007 | [+0.631, +5.281] |
| var_keypoint_contact | ok  | 34.2 | +0.119 | [+0.009, +0.233] | +1.513 | [-0.998, +3.883] |
| var_keypoint_contact_it11 | ok  | 28.1 | +0.060 | [-0.071, +0.186] | +1.646 | [-0.777, +3.975] |
| omniretarget | ok  | 42.0 | +0.000 |  | +0.000 |  |
| var_no_mesh | ok  | 31.0 | -0.429 | [-0.574, -0.298] | -2.421 | [-5.488, +0.320] |
| sc_constant_pose | ok  | 1.1 | -6.177 | [-6.556, -5.805] | -27272.734 | [-53881.482, -6019.821] |
| keypoint_ik | ok  | 21.4 | -8.100 | [-9.386, -6.826] | -136059.539 | [-211945.957, -71696.287] |

Per-experiment metric means:

**object_interaction_kinematics**

| method | penetration_duration | max_penetration_cm | foot_skating_duration | max_foot_skating_cm_s | contact_preservation | spurious_contact_duration | pelvis_path_error_cm | motion_velocity_error_cm_s |
|---|---|---|---|---|---|---|---|---|
| keypoint_ik | 0.3685 | 1.1667 | 0.1450 | 6.0042 | 0.7530 | 0.0000 | 0.2265 | 0.6752 |
| omniretarget | 0.0000 | 0.0000 | 0.0022 | 0.6234 | 0.9840 | 0.0017 | 0.7623 | 3.1504 |
| sc_constant_pose | 0.0181 | 0.2304 | 0.0000 | 0.0000 | 0.4126 | 1.0000 | 23.3622 | 22.3066 |
| sc_select_best | 0.0000 | 0.0000 | 0.0000 | 0.2186 | 1.0000 | 0.0017 | 0.4868 | 3.0872 |
| var_contact_aware | 0.0000 | 0.0000 | 0.0017 | 0.4377 | 1.0000 | 0.0017 | 0.5852 | 2.9680 |
| var_keypoint_contact | 0.0000 | 0.0000 | 0.0008 | 0.4898 | 1.0000 | 0.0033 | 0.3947 | 3.4216 |
| var_keypoint_contact_it11 | 0.0000 | 0.0000 | 0.0009 | 0.5008 | 1.0000 | 0.0033 | 0.3798 | 3.4202 |
| var_keypoint_contact_w100 | 0.0000 | 0.0000 | 0.0000 | 0.4072 | 1.0000 | 0.0033 | 0.3922 | 3.5335 |
| var_no_mesh | 0.0000 | 0.0000 | 0.0005 | 0.3518 | 0.9628 | 0.0048 | 0.5316 | 3.6296 |

**terrain_interaction_kinematics**

| method | penetration_duration | max_penetration_cm | foot_skating_duration | max_foot_skating_cm_s | contact_preservation | spurious_contact_duration | pelvis_path_error_cm | motion_velocity_error_cm_s |
|---|---|---|---|---|---|---|---|---|
| keypoint_ik | 0.1944 | 1.1054 | 0.2234 | 21.0755 | 0.9959 | 0.0003 | 0.3208 | 2.6375 |
| omniretarget | 0.0000 | 0.0058 | 0.0013 | 0.5090 | 0.9893 | 0.0008 | 0.7587 | 7.5536 |
| sc_constant_pose | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8201 | 0.1976 | 30.4475 | 30.2224 |
| sc_select_best | 0.0000 | 0.0118 | 0.0000 | 0.2396 | 1.0000 | 0.0022 | 0.5838 | 7.0294 |
| var_contact_aware | 0.0000 | 0.0059 | 0.0022 | 0.5692 | 1.0000 | 0.0008 | 0.7108 | 7.9003 |
| var_keypoint_contact | 0.0000 | 0.0118 | 0.0017 | 0.6153 | 1.0000 | 0.0028 | 0.5798 | 7.0199 |
| var_keypoint_contact_it11 | 0.0000 | 0.0119 | 0.0012 | 0.5986 | 1.0000 | 0.0028 | 0.5819 | 7.0217 |
| var_keypoint_contact_w100 | 0.0000 | 0.0117 | 0.0000 | 0.2906 | 1.0000 | 0.0028 | 0.5568 | 7.1962 |
| var_no_mesh | 0.0000 | 0.0109 | 0.0034 | 0.8478 | 0.9900 | 0.0028 | 0.6390 | 6.8095 |

**augmented_interaction_kinematics**

| method | penetration_duration | max_penetration_cm | foot_skating_duration | max_foot_skating_cm_s | contact_preservation | spurious_contact_duration | pelvis_path_error_cm | motion_velocity_error_cm_s |
|---|---|---|---|---|---|---|---|---|
| keypoint_ik | 0.6435 | 6.5716 | 0.1369 | 15.4286 | 0.5260 | 0.0046 | 1.7311 | 1.5322 |
| omniretarget | 0.0018 | 0.0915 | 0.0040 | 0.7943 | 0.9554 | 0.0069 | 2.3374 | 7.7403 |
| sc_constant_pose | 0.0172 | 0.2447 | 0.0000 | 0.0000 | 0.6022 | 0.6025 | 27.1933 | 26.8209 |
| sc_select_best | 0.0000 | 0.0177 | 0.0008 | 0.4068 | 1.0000 | 0.0034 | 2.0339 | 8.1862 |
| var_contact_aware | 0.0000 | 0.0105 | 0.0031 | 0.6676 | 1.0000 | 0.0069 | 2.1979 | 7.9843 |
| var_keypoint_contact | 0.0027 | 0.2632 | 0.0009 | 0.5571 | 0.9750 | 0.0213 | 1.8116 | 7.9639 |
| var_keypoint_contact_it11 | 0.0027 | 0.2641 | 0.0021 | 0.4536 | 0.9376 | 0.0228 | 1.8831 | 7.5177 |
| var_keypoint_contact_w100 | 0.0027 | 0.2443 | 0.0009 | 0.4166 | 1.0000 | 0.0213 | 1.7874 | 8.6829 |
| var_no_mesh | 0.1015 | 0.7876 | 0.0008 | 0.5549 | 0.7987 | 0.0182 | 1.9725 | 7.0901 |

