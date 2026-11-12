# Experiments

All experiments score a robot joint trajectory per demonstration; every number
is recomputed from that trajectory. A method that fails any instance, or leaves
the robot's joint or velocity limits, receives the worst value of every metric
in every experiment.

## object_interaction_kinematics (weight 0.35)

Establishes whether a method can carry a held object for a demonstrator of a
different size: both hands must stay on the box sides, no body part may enter
the box or the ground, planted feet must neither slide nor leave the ground, and
the motion should follow the demonstration. Measures the constraint side
(penetration, skating) and the interaction side (contact preservation) together
with motion fidelity.

## terrain_interaction_kinematics (weight 0.35)

Same question for scene contact: stepping up a ramp onto a platform and bracing
a hand on a tall column, with terrain penetration, foot skating, contact of the
bracing hand and planted feet, and motion fidelity measured.

## augmented_interaction_kinematics (weight 0.30)

Establishes whether demonstrated interactions transfer to a modified scene: the
robot acts in a scene where the box has been resized and starts displaced, or
where the platform and column heights differ from what the demonstrator used.
Contacts are judged against the modified scene, so copying the demonstrated
motion is not enough.
