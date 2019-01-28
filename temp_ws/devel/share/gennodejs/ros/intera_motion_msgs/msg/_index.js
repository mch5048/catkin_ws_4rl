
"use strict";

let WaypointOptions = require('./WaypointOptions.js');
let MotionStatus = require('./MotionStatus.js');
let WaypointSimple = require('./WaypointSimple.js');
let Waypoint = require('./Waypoint.js');
let InterpolatedPath = require('./InterpolatedPath.js');
let TrajectoryAnalysis = require('./TrajectoryAnalysis.js');
let TrackingOptions = require('./TrackingOptions.js');
let TrajectoryOptions = require('./TrajectoryOptions.js');
let JointTrackingError = require('./JointTrackingError.js');
let Trajectory = require('./Trajectory.js');
let EndpointTrackingError = require('./EndpointTrackingError.js');
let MotionCommandAction = require('./MotionCommandAction.js');
let MotionCommandActionFeedback = require('./MotionCommandActionFeedback.js');
let MotionCommandActionResult = require('./MotionCommandActionResult.js');
let MotionCommandFeedback = require('./MotionCommandFeedback.js');
let MotionCommandGoal = require('./MotionCommandGoal.js');
let MotionCommandResult = require('./MotionCommandResult.js');
let MotionCommandActionGoal = require('./MotionCommandActionGoal.js');

module.exports = {
  WaypointOptions: WaypointOptions,
  MotionStatus: MotionStatus,
  WaypointSimple: WaypointSimple,
  Waypoint: Waypoint,
  InterpolatedPath: InterpolatedPath,
  TrajectoryAnalysis: TrajectoryAnalysis,
  TrackingOptions: TrackingOptions,
  TrajectoryOptions: TrajectoryOptions,
  JointTrackingError: JointTrackingError,
  Trajectory: Trajectory,
  EndpointTrackingError: EndpointTrackingError,
  MotionCommandAction: MotionCommandAction,
  MotionCommandActionFeedback: MotionCommandActionFeedback,
  MotionCommandActionResult: MotionCommandActionResult,
  MotionCommandFeedback: MotionCommandFeedback,
  MotionCommandGoal: MotionCommandGoal,
  MotionCommandResult: MotionCommandResult,
  MotionCommandActionGoal: MotionCommandActionGoal,
};
