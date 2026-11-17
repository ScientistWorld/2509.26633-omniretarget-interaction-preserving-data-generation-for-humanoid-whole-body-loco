# Experiments

All three experiments use real human motion, a 3D humanoid and scene collision geometry. They jointly measure interaction preservation, penetration, skating, support failure, unwanted contact, pelvis-path error and body-pose error. Every experiment has the same nine scored metrics. Failure anywhere assigns the worst value everywhere.

- `object_interaction_kinematics` (0.35): Can a differently proportioned humanoid manipulate a physical box without losing demonstrated hand contact, penetrating the box or ground, sliding or floating its planted feet, or abandoning the demonstrated motion? Test includes 48 windows from unseen demonstrators.
- `terrain_interaction_kinematics` (0.35): Can the humanoid preserve climbing interactions and motion while respecting terrain clearance and stable foot placement? Test includes 14 windows from three held-out recordings.
- `augmented_interaction_kinematics` (0.30): Can demonstrated interactions transfer when box dimensions and initial poses or platform heights change? Test includes 24 box windows and 14 climbing windows. This exposes errors that copying nominal-scene motion cannot resolve.

The benchmark evaluates kinematic quality, not downstream control success or physical stability. Test results are computed from raw trajectories; there are no agreement-with-paper metrics.
