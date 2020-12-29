#!/usr/bin/env python

import collections
import time
try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk
import tkFileDialog

try:
    import rospy
    import actionlib

    import ibmmpy.msg
except ImportError:
    rospy = None


FIXATION_DETECTOR_CONFIG_NAME = "fixation_detector"


def get_latest_and_clear(queue):
    res = None
    try:
        while True:
            res = queue.popleft()
    except IndexError:
        return res

class FixationDetectorControllerFrame(tk.Frame, object):
    def __init__(self, parent, initial_config={}):
        super(FixationDetectorControllerFrame, self).__init__(parent)

        initial_config = initial_config.get(FIXATION_DETECTOR_CONFIG_NAME, {})

        # top level stuff
        self._config_frame = tk.Frame(self)
        self._config_frame.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self._config_frame.columnconfigure(0, weight=1)

        self._status_frame = tk.Frame(self, bd=2, relief=tk.GROOVE)
        self._status_frame.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        # basic info
        self._basic_frame = tk.Frame(self._config_frame, bd=2, relief=tk.GROOVE)
        self._basic_topic_var = tk.StringVar()
        self._basic_topic_var.set(initial_config.get("topic", ""))
        self._basic_topic_label = tk.Label(self._basic_frame, text="Topic:")
        self._basic_topic_label.grid(row=0, column=0, sticky='w')
        self._basic_topic_entry = tk.Entry(self._basic_frame, textvariable=self._basic_topic_var)
        self._basic_topic_entry.grid(row=0, column=1, sticky='new')
        self._basic_topic_entry.bind("<Return>", func=lambda _: self._connect())
        self._basic_connect_btn = tk.Button(self._basic_frame, text="Connect", command=self._connect)
        self._basic_connect_btn.grid(row=1, column=1, sticky="ne")
        self._basic_gaze_topic_var = tk.StringVar()
        self._basic_gaze_topic_var.set(initial_config.get("gaze_topic", ""))
        self._basic_gaze_topic_label = tk.Label(self._basic_frame, text="Gaze topic:")
        self._basic_gaze_topic_label.grid(row=2, column=0, sticky="nw")
        self._basic_gaze_topic_entry = tk.Entry(self._basic_frame, textvariable=self._basic_gaze_topic_var)
        self._basic_gaze_topic_entry.grid(row=2, column=1, sticky="new")

        self._basic_use_world_var = tk.IntVar()
        self._basic_use_world_var.set(initial_config.get("use_world", 1))
        self._basic_use_world = tk.Checkbutton(self._basic_frame, text="Use world", variable=self._basic_use_world_var)
        self._basic_use_world.grid(row=3, column=0, sticky="nw")
        self._basic_use_eye0_var = tk.IntVar()
        self._basic_use_eye0_var.set(initial_config.get("use_eye0", 0))
        self._basic_use_eye0 = tk.Checkbutton(self._basic_frame, text="Use eye 0", variable=self._basic_use_eye0_var)
        self._basic_use_eye0.grid(row=3, column=1, sticky="nw")
        self._basic_use_eye1_var = tk.IntVar()
        self._basic_use_eye1_var.set(initial_config.get("use_eye1", 0))
        self._basic_use_eye1 = tk.Checkbutton(self._basic_frame, text="Use eye 1", variable=self._basic_use_eye1_var)
        self._basic_use_eye1.grid(row=4, column=1, sticky="nw")

        self._basic_combination_period_label = tk.Label(self._basic_frame, text="Label combination period (s):")
        self._basic_combination_period_label.grid(row=5, column=0, sticky="nw")
        self._basic_combination_period_var = tk.StringVar()
        self._basic_combination_period_var.set(str(initial_config.get("label_combination_period", 0.033)))
        self._basic_combination_period_entry = tk.Entry(self._basic_frame, textvariable=self._basic_combination_period_var)  # TODO: validate
        self._basic_combination_period_entry.grid(row=5, column=1, sticky="new")

        self._basic_min_fix_dur_label = tk.Label(self._basic_frame, text="Minimum fixation duration (ms):")
        self._basic_min_fix_dur_label.grid(row=6, column=0, sticky="nw")
        self._basic_min_fix_dur_var = tk.StringVar()
        self._basic_min_fix_dur_var.set(str(initial_config.get("min_fix_duration", 100)))
        self._basic_min_fix_dur_entry = tk.Entry(self._basic_frame, textvariable=self._basic_min_fix_dur_var)  # TODO: validate
        self._basic_min_fix_dur_entry.grid(row=6, column=1, sticky="new")


        self._basic_frame.columnconfigure(1, weight=1)
        self._basic_frame.grid(row=0, column=0, sticky="nsew")

        # calibration loading
        self._load_cal_frame = tk.Frame(self._config_frame, bd=2, relief=tk.GROOVE)
        self._cal_load_var = tk.StringVar()
        self._cal_load_title = tk.Label(self._load_cal_frame, text="Load calibration from:")
        self._cal_load_title.grid(row=0, column=0, sticky="nw")
        self._cal_load_label = tk.Label(self._load_cal_frame, textvariable=self._cal_load_var)
        self._cal_load_label.grid(row=1, column=0, sticky='new')
        self._cal_load_btn = tk.Button(self._load_cal_frame, text="Choose", command=self._set_cal_load_var)
        self._cal_load_btn.grid(row=0, column=1, sticky='nw')
        self._cal_run_load_btn = tk.Button(self._load_cal_frame, text="Load calibration", command=self._load_cal)
        self._cal_run_load_btn.grid(row=2, column=0, columnspan=2, sticky='n')
        
        self._load_cal_frame.rowconfigure(1, weight=1)
        self._load_cal_frame.columnconfigure(0, weight=1)
        self._load_cal_frame.grid(row=1, column=0, sticky="nsew")

        # calibration running
        self._run_cal_frame = tk.Frame(self._config_frame, bd=2, relief=tk.GROOVE)
        self._run_cal_label = tk.Label(self._run_cal_frame, text="Run calibration")
        self._run_cal_label.grid(row=0, column=0, columnspan=2, sticky="nw")
        self._run_cal_start_btn = tk.Button(self._run_cal_frame, text="Start", command=self._start_cal)
        self._run_cal_stop_btn = tk.Button(self._run_cal_frame, text="Stop", command=self._stop_cal)
        self._run_cal_start_btn.grid(row=1, column=0, sticky='n')
        self._run_cal_stop_btn.grid(row=1, column=1, sticky='n')

        self._run_cal_frame.columnconfigure(0, weight=1)
        self._run_cal_frame.columnconfigure(1, weight=1)
        self._run_cal_frame.grid(row=2, column=0, sticky="nsew")

        # detector running
        self._run_detector_frame = tk.Frame(self._config_frame, bd=2, relief=tk.GROOVE)
        self._run_detector_label = tk.Label(self._run_detector_frame, text="Run detector")
        self._run_detector_label.grid(row=0, column=0, columnspan=2, sticky="nw")
        self._run_detector_start_btn = tk.Button(self._run_detector_frame, text="Start", command=self._start_detection)
        self._run_detector_stop_btn = tk.Button(self._run_detector_frame, text="Stop", command=self._stop_detection)
        self._run_detector_start_btn.grid(row=1, column=0, sticky='n')
        self._run_detector_stop_btn.grid(row=1, column=1, sticky='n')

        self._run_detector_frame.columnconfigure(0, weight=1)
        self._run_detector_frame.columnconfigure(1, weight=1)
        self._run_detector_frame.grid(row=3, column=0, sticky="nsew")

        # information
        self._status_label = tk.Label(self._status_frame, text="Status:")
        self._status_label.grid(row=0, column=0, sticky="nw")
        self._status_val = tk.StringVar()
        self._status_val.set("Not connected")
        self._status_val_label = tk.Label(self._status_frame, textvariable=self._status_val)
        self._status_val_label.grid(row=0, column=1, sticky="ne")

        self._goal_label = tk.Label(self._status_frame, text="Goal:")
        self._goal_label.grid(row=1, column=0, sticky="nw")
        self._goal_val = tk.StringVar()
        self._goal_val_label = tk.Label(self._status_frame, textvariable=self._goal_val, justify=tk.LEFT)
        self._goal_val_label.grid(row=2, column=1, sticky="ne")

        self._goal_status_label = tk.Label(self._status_frame, text="Goal status:")
        self._goal_status_label.grid(row=3, column=0, sticky="nw")
        self._goal_status_val = tk.StringVar()
        self._goal_status_val_label = tk.Label(self._status_frame, textvariable=self._goal_status_val)
        self._goal_status_val_label.grid(row=3, column=1, sticky="ne")

        self._result_label = tk.Label(self._status_frame, text="Result:")
        self._result_label.grid(row=4, column=0, sticky="nw")
        self._result_var = tk.StringVar()
        self._result_var_label = tk.Label(self._status_frame, textvariable=self._result_var)
        self._result_var_label.grid(row=4, column=1, sticky="ne")

        self._feedback_label = tk.Label(self._status_frame, text="Feedback:")
        self._feedback_label.grid(row=5, column=0, sticky="nw")
        self._feedback_var = tk.StringVar()
        self._feedback_var_label = tk.Label(self._status_frame, textvariable=self._feedback_var)
        self._feedback_var_label.grid(row=5, column=1, sticky="ne")

        # set up action client
        self._action_client = None
        self._connect_timeout_time = None
        self._connect()

        self._status_queue = collections.deque()
        self._feedback_queue = collections.deque()
        self._result_queue = collections.deque()

        # for external use
        self.log_dir = ""

    def _connect(self):
        topic = self._basic_topic_var.get()
        if topic and rospy:
            rospy.loginfo("Connecting to {}...".format(topic))
            self._action_client = actionlib.SimpleActionClient(topic + "/detect", ibmmpy.msg.DetectorAction)
            self._status_val.set("Connecting...")
            self._connect_timeout_time = rospy.get_rostime() + rospy.Duration(2.)
            self.after(100, self._check_connected)
        else:
            self._set_disconnected()

    def _check_connected(self):
        if self._action_client and self._action_client.wait_for_server(rospy.Duration(0.001)):
            self._set_connected()
        elif self._connect_timeout_time and rospy.get_rostime() >= self._connect_timeout_time:
            self._set_disconnected()
        else:
            self.after(100, self._check_connected)

    def _set_connected(self):
        self._status_val.set("Connected")
        self._cal_run_load_btn.configure(state=tk.NORMAL)
        self._run_cal_start_btn.configure(state=tk.NORMAL)
        self._run_cal_stop_btn.configure(state=tk.NORMAL)
        self._run_detector_start_btn.configure(state=tk.NORMAL)
        self._run_detector_stop_btn.configure(state=tk.NORMAL)

    def _set_disconnected(self):
        rospy.loginfo("Not connected")
        self._status_val.set("Not connected")
        self._cal_run_load_btn.configure(state=tk.DISABLED)
        self._run_cal_start_btn.configure(state=tk.DISABLED)
        self._run_cal_stop_btn.configure(state=tk.DISABLED)
        self._run_detector_start_btn.configure(state=tk.DISABLED)
        self._run_detector_stop_btn.configure(state=tk.DISABLED)

    def _send_goal(self, goal):
        # load in from basic stuff
        goal.topic = self._basic_gaze_topic_var.get()
        goal.use_world = self._basic_use_world_var.get()
        goal.use_eye0 = self._basic_use_eye0_var.get()
        goal.use_eye1 = self._basic_use_eye1_var.get()
        try:
            goal.label_combination_period = float(self._basic_combination_period_var.get())
        except ValueError:
            rospy.logerror("Label combination must be a float")
            return
        try:
            goal.min_fix_duration = float(self._basic_min_fix_dur_var.get())
        except ValueError:
            rospy.logerror("Label combination must be a float")
            return

        if self._action_client:
            self._result_queue.append("")
            self._feedback_queue.append("")

            self._goal_val.set(str(goal))
            self._status_queue.append(actionlib.simple_action_client.SimpleGoalState.PENDING)
            self._action_client.send_goal(goal, self._recv_done, self._recv_active, self._recv_feedback)
    
    def _recv_done(self, state, result):
        self._status_queue.append(state)
        self._result_queue.append(self._action_client.get_goal_status_text())
        self._action_client.stop_tracking_goal()

    def _recv_active(self):
        self._status_queue.append(self._action_client.get_state())

    def _recv_feedback(self, feedback):
        self._feedback_queue.append(feedback)

    def _set_cal_load_var(self):
        self._cal_load_var.set(tkFileDialog.askdirectory(title="Select location to load calibration files from"))

    def _load_cal(self):
        if self._action_client:
            if not self._cal_load_var.get():
                rospy.logerror("Must specify a directory for calibration")
                return
            goal = ibmmpy.msg.DetectorGoal()
            goal.action = ibmmpy.msg.DetectorGoal.ACTION_CALIBRATE
            goal.load_dir = self._cal_load_var.get()
            self._send_goal(goal)

    def _start_cal(self):
        if self._action_client:
            goal = ibmmpy.msg.DetectorGoal()
            goal.action = ibmmpy.msg.DetectorGoal.ACTION_CALIBRATE
            goal.log_dir = self.log_dir
            self._send_goal(goal)

    def _stop_cal(self):
        if self._action_client:
            self._action_client.cancel_goal()

    def _start_detection(self):
        if self._action_client:
            goal = ibmmpy.msg.DetectorGoal()
            goal.action = ibmmpy.msg.DetectorGoal.ACTION_DETECT
            self._send_goal(goal)

    def _stop_detection(self):
        if self._action_client:
            self._action_client.cancel_goal()

    def run_once(self):
        self.update()

        status = get_latest_and_clear(self._status_queue)
        if status is not None:
            self._goal_status_val.set(actionlib.GoalStatus.to_string(status))
        
        feedback = get_latest_and_clear(self._feedback_queue)
        if feedback is not None:
            self._feedback_var.set(str(feedback))

        result = get_latest_and_clear(self._result_queue)
        if result is not None:
            self._result_var.set(str(result))

    def get_config(self):
        return { FIXATION_DETECTOR_CONFIG_NAME: {
            "topic": self._basic_topic_var.get(),
            "gaze_topic": self._basic_gaze_topic_var.get(),
            "use_world": bool(self._basic_use_world_var.get()),
            "use_eye0": bool(self._basic_use_eye0_var.get()),
            "use_eye1": bool(self._basic_use_eye1_var.get()),
            "label_combination_period": _float_or_zero(self._basic_combination_period_var.get()),
            "min_fix_duration": _float_or_zero(self._basic_min_fix_dur_var.get())
        } }

def _float_or_zero(s):
    try:
        return float(s)
    except ValueError:
        return 0.

if __name__ == "__main__":
    if rospy:
        rospy.init_node("fixation_controller", anonymous=True)
    parent = tk.Tk()
    controller = FixationDetectorControllerFrame(parent)
    controller.pack(expand=True, fill=tk.BOTH)
    
    while rospy is None or not rospy.is_shutdown():
        controller.run_once()
        time.sleep(0.01)
