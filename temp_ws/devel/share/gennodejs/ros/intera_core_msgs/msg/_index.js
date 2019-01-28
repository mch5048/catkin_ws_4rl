
"use strict";

let CameraControl = require('./CameraControl.js');
let IOStatus = require('./IOStatus.js');
let DigitalIOStates = require('./DigitalIOStates.js');
let AnalogIOState = require('./AnalogIOState.js');
let CollisionDetectionState = require('./CollisionDetectionState.js');
let HomingCommand = require('./HomingCommand.js');
let JointCommand = require('./JointCommand.js');
let NavigatorState = require('./NavigatorState.js');
let IONodeConfiguration = require('./IONodeConfiguration.js');
let IOComponentConfiguration = require('./IOComponentConfiguration.js');
let IOComponentCommand = require('./IOComponentCommand.js');
let EndpointState = require('./EndpointState.js');
let EndpointStates = require('./EndpointStates.js');
let EndpointNamesArray = require('./EndpointNamesArray.js');
let IOComponentStatus = require('./IOComponentStatus.js');
let HeadPanCommand = require('./HeadPanCommand.js');
let IODeviceStatus = require('./IODeviceStatus.js');
let IODataStatus = require('./IODataStatus.js');
let URDFConfiguration = require('./URDFConfiguration.js');
let CollisionAvoidanceState = require('./CollisionAvoidanceState.js');
let InteractionControlCommand = require('./InteractionControlCommand.js');
let InteractionControlState = require('./InteractionControlState.js');
let HeadState = require('./HeadState.js');
let IONodeStatus = require('./IONodeStatus.js');
let IODeviceConfiguration = require('./IODeviceConfiguration.js');
let AnalogIOStates = require('./AnalogIOStates.js');
let RobotAssemblyState = require('./RobotAssemblyState.js');
let SEAJointState = require('./SEAJointState.js');
let CameraSettings = require('./CameraSettings.js');
let NavigatorStates = require('./NavigatorStates.js');
let DigitalOutputCommand = require('./DigitalOutputCommand.js');
let JointLimits = require('./JointLimits.js');
let DigitalIOState = require('./DigitalIOState.js');
let AnalogOutputCommand = require('./AnalogOutputCommand.js');
let HomingState = require('./HomingState.js');
let CalibrationCommandActionGoal = require('./CalibrationCommandActionGoal.js');
let CalibrationCommandActionResult = require('./CalibrationCommandActionResult.js');
let CalibrationCommandFeedback = require('./CalibrationCommandFeedback.js');
let CalibrationCommandGoal = require('./CalibrationCommandGoal.js');
let CalibrationCommandActionFeedback = require('./CalibrationCommandActionFeedback.js');
let CalibrationCommandResult = require('./CalibrationCommandResult.js');
let CalibrationCommandAction = require('./CalibrationCommandAction.js');

module.exports = {
  CameraControl: CameraControl,
  IOStatus: IOStatus,
  DigitalIOStates: DigitalIOStates,
  AnalogIOState: AnalogIOState,
  CollisionDetectionState: CollisionDetectionState,
  HomingCommand: HomingCommand,
  JointCommand: JointCommand,
  NavigatorState: NavigatorState,
  IONodeConfiguration: IONodeConfiguration,
  IOComponentConfiguration: IOComponentConfiguration,
  IOComponentCommand: IOComponentCommand,
  EndpointState: EndpointState,
  EndpointStates: EndpointStates,
  EndpointNamesArray: EndpointNamesArray,
  IOComponentStatus: IOComponentStatus,
  HeadPanCommand: HeadPanCommand,
  IODeviceStatus: IODeviceStatus,
  IODataStatus: IODataStatus,
  URDFConfiguration: URDFConfiguration,
  CollisionAvoidanceState: CollisionAvoidanceState,
  InteractionControlCommand: InteractionControlCommand,
  InteractionControlState: InteractionControlState,
  HeadState: HeadState,
  IONodeStatus: IONodeStatus,
  IODeviceConfiguration: IODeviceConfiguration,
  AnalogIOStates: AnalogIOStates,
  RobotAssemblyState: RobotAssemblyState,
  SEAJointState: SEAJointState,
  CameraSettings: CameraSettings,
  NavigatorStates: NavigatorStates,
  DigitalOutputCommand: DigitalOutputCommand,
  JointLimits: JointLimits,
  DigitalIOState: DigitalIOState,
  AnalogOutputCommand: AnalogOutputCommand,
  HomingState: HomingState,
  CalibrationCommandActionGoal: CalibrationCommandActionGoal,
  CalibrationCommandActionResult: CalibrationCommandActionResult,
  CalibrationCommandFeedback: CalibrationCommandFeedback,
  CalibrationCommandGoal: CalibrationCommandGoal,
  CalibrationCommandActionFeedback: CalibrationCommandActionFeedback,
  CalibrationCommandResult: CalibrationCommandResult,
  CalibrationCommandAction: CalibrationCommandAction,
};
