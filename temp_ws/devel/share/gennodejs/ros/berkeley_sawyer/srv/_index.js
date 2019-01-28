
"use strict";

let get_kinectdata = require('./get_kinectdata.js')
let save_kinectdata = require('./save_kinectdata.js')
let delete_traj = require('./delete_traj.js')
let init_traj = require('./init_traj.js')
let init_traj_visualmpc = require('./init_traj_visualmpc.js')
let get_action = require('./get_action.js')

module.exports = {
  get_kinectdata: get_kinectdata,
  save_kinectdata: save_kinectdata,
  delete_traj: delete_traj,
  init_traj: init_traj,
  init_traj_visualmpc: init_traj_visualmpc,
  get_action: get_action,
};
