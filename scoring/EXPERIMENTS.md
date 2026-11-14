# Experiments

All experiments score a robot joint trajectory per demonstration; every number
is recomputed from that trajectory. A method that fails any instance, exceeds
its counted CPU budget, or leaves the robot's joint or velocity limits receives
the worst value of every metric in every experiment.

## object_interaction_kinematics (weight 0.35)

Establishes whether a method can carry a held object for a demonstrator of a
different size and build while also negotiating terrain (a ramp up to a
platform): both hands must stay on the box sides, no body part may enter the box
or the ground, planted feet must neither slide nor leave the ground, the robot
must not touch surfaces the demonstrator kept clear of, and the motion should
follow the demonstration at the robot's scale. Measures the constraint side
(penetration, skating, spurious contact) and the interaction side (contact
preservation) together with motion fidelity.

## terrain_interaction_kinematics (weight 0.35)

Same question for scene contact: stepping up a ramp onto a platform and bracing
a hand on a tall column, with terrain penetration, foot skating, spurious
contact, contact of the bracing hand and planted feet, and motion fidelity
measured.

## augmented_interaction_kinematics (weight 0.30)

Establishes whether demonstrated interactions transfer to a modified scene: the
robot acts in a scene where the box has been resized and starts displaced, or
where the platform and column heights differ from the nominal scene. Contacts,
clearances and the pelvis height are judged against the modified scene, so
copying the demonstrated motion is not enough.
