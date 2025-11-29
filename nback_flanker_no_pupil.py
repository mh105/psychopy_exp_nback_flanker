#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.2a1),
    on Fri Nov 28 20:28:21 2025
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '4'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from eeg
import pyxid2
import threading
import signal


def exit_after(s):
    '''
    function decorator to raise KeyboardInterrupt exception
    if function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, signal.raise_signal, args=[signal.SIGINT])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer


@exit_after(1)  # exit if function takes longer than 1 seconds
def _get_xid_devices():
    return pyxid2.get_xid_devices()


def get_xid_devices():
    print("Getting a list of all attached XID devices...")
    attempt_count = 0
    while attempt_count >= 0:
        attempt_count += 1
        print('     Attempt:', attempt_count)
        attempt_count *= -1  # try to exit the while loop
        try:
            devices = _get_xid_devices()
        except KeyboardInterrupt:
            attempt_count *= -1  # get back in the while loop
    return devices


devices = get_xid_devices()

if devices:
    dev = devices[0]
    print("Found device:", dev)
    assert dev.device_name in ['Cedrus C-POD', 'Cedrus StimTracker Quad'], "Incorrect XID device detected."
    if dev.device_name == 'Cedrus C-POD':
        pod_name = 'C-POD'
    else:
        pod_name = 'M-POD'
    dev.set_pulse_duration(50)  # set pulse duration to 50ms

    # Start EEG recording
    print("Sending trigger code 126 to start EEG recording...")
    dev.activate_line(bitmask=126)  # trigger 126 will start EEG
    print("Waiting 10 seconds for the EEG recording to start...\n")
    core.wait(10)  # wait 10s for the EEG system to start recording

    # Marching lights test
    print(f"{pod_name}<->eego 7-bit trigger lines test...")
    for line in range(1, 8):  # raise lines 1-7 one at a time
        print("  raising line {} (bitmask {})".format(line, 2 ** (line-1)))
        dev.activate_line(lines=line)
        core.wait(0.5)  # wait 500ms between two consecutive triggers
    dev.con.set_digio_lines_to_mask(0)  # XidDevice.clear_all_lines()
    print("EEG system is now ready for the experiment to start.\n")

else:
    # Dummy XidDevice for code components to run without C-POD connected
    class dummyXidDevice(object):
        def __init__(self):
            pass
        def activate_line(self, lines=None, bitmask=None):
            pass


    print("WARNING: No C/M-POD connected for this session! "
          "You must start/stop EEG recording manually!\n")
    dev = dummyXidDevice()

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.2a1'
expName = 'nback_flanker_no_pupil'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s/%s_%s_%s' % (expInfo['participant'], expInfo['participant'], expName, expInfo['session'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/alexhe/Library/CloudStorage/Dropbox/Active_projects/PsychoPy/exp_nback_flanker/nback_flanker_no_pupil.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # log the filename of last_app_load.log
    print('target_last_app_load_log_file: ' + filename + '_last_app_load.log')
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('debug')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('welcomeKey') is None:
        # initialise welcomeKey
        welcomeKey = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='welcomeKey',
        )
    if deviceManager.getDevice('pracInstructKey') is None:
        # initialise pracInstructKey
        pracInstructKey = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='pracInstructKey',
        )
    if deviceManager.getDevice('earlyResp') is None:
        # initialise earlyResp
        earlyResp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='earlyResp',
        )
    if deviceManager.getDevice('resp') is None:
        # initialise resp
        resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='resp',
        )
    if deviceManager.getDevice('instructKey') is None:
        # initialise instructKey
        instructKey = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instructKey',
        )
    # create speaker 'read_thank_you'
    deviceManager.addDevice(
        deviceName='read_thank_you',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "__start__" ---
    # Run 'Begin Experiment' code from trigger_table
    ##TASK ID TRIGGER VALUES##
    # special code 100 (task start, task ID should follow immediately)
    task_start_code = 100
    # special code 108 (task ID for nback flanker task)
    task_ID_code = 108
    print("Starting experiment: < N-back Flanker Task >. Task ID:", task_ID_code)
    
    ##GENERAL TRIGGER VALUES##
    # special code 122 (block start)
    block_start_code = 122
    # special code 123 (block end)
    block_end_code = 123
    
    ##TASK SPECIFIC TRIGGER VALUES##
    # N.B.: only use values 1-99 and provide clear comments on used values
    iti_start_code = 9
    letter_start_code = 10
    cue_start_code = 11
    feedback_start_code = 12
    
    # Run 'Begin Experiment' code from task_id
    dev.activate_line(bitmask=task_start_code)  # special code for task start
    core.wait(0.5)  # wait 500ms between two consecutive triggers
    dev.activate_line(bitmask=task_ID_code)  # special code for task ID
    
    # Run 'Begin Experiment' code from condition_setup
    """
    Congruency is defined at the response level depending on the N-back task correct response.
    
    0-Back trial types:
        A) Correct response = NO (C_t != H)
            A1 EASY  (congruent):   F_t ≠ C_t
            A2 HARD  (incongruent): F_t = C_t
    
        B) Correct response = YES (C_t = H)
            B1 EASY  (congruent):   F_t = C_t
            B2 HARD  (incongruent): F_t ≠ C_t
    
    1-Back trial types:
        A) Correct response = NO (C_t != C_{t-1})
            A1 EASY  (congruent):   F_t ≠ C_t, F_t ≠ C_{t-1}, F_t ≠ F_{t-1}
            A2 HARD  (incongruent): F_t = C_t  and  F_t = F_{t-1}
    
        B) Correct response = YES (C_t = C_{t-1})
            B1 EASY  (congruent):   F_t = C_t  (== C_{t-1})
            B2 HARD  (incongruent): F_t ≠ C_t  and  F_t ≠ F_{t-1}
    
    Implementation notes:
    - We generate the entire 1-back block using a seed previous state (C_{-1}, F_{-1}).
    - A2 is only feasible if F_{t-1} != C_{t-1}. The scheduler always chooses from remaining feasible types; if A2 isn't
      feasible at a step, it is deferred until it becomes feasible later.
    - We distribute trial types evenly: base = n_trials // 4 for each trial type
    """
    
    # single RNG for this session
    rng = np.random.default_rng()
    
    # ------------------------------------------------------
    # Letters used in the task
    # ------------------------------------------------------
    letters = ['S', 'H', 'C', 'F']
    target0 = 'H'  # target letter in the 0-back block
    
    # ------------------------------------------------------
    # Visual stimuli durations
    # ------------------------------------------------------
    stimulusDur = 0.5  # seconds
    cueDur = 3.0       # seconds
    respCueJitter = [1.5, 3.5]  # seconds
    
    # ------------------------------------------------------
    # Numbers of blocks and trials
    # ------------------------------------------------------
    n_back_blocks = [0, 1]  # 0-back first, then 1-back
    n_practice = 20
    n_main = 120
    
    
    # ------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------
    def pick_not_in(excluded, rng, letters=letters):
        """
        pick a random letter NOT in a set
        """
        candidates = [letter for letter in letters if letter not in excluded]
        return rng.choice(candidates)
    
    
    def other(x, rng, letters=letters):
        """
        pick any letter except x
        """
        candidates = [letter for letter in letters if letter != x]
        return rng.choice(candidates)
    
    
    def _type_counts(n_trials):
        types = ['A1', 'A2', 'B1', 'B2']
        base = n_trials // 4
        rem = n_trials % 4
        assert rem == 0, "n_trials must be multiple of 4 for balancing."
        counts = {b: base for b in types}
        return counts
    
    
    # ======================================================
    # 0-BACK
    # ======================================================
    def make_0back(n_trials, target, rng):
        """
        make_0back():
            Builds a FULLY BALANCED 0-back block across the four trial types (A1, A2, B1, B2)
        """
        counts = _type_counts(n_trials)  # dict: {'A1': n, 'A2': n, 'B1': n, 'B2': n}
        rows = []
    
        for b in counts.keys():
            for _ in range(counts[b]):
    
                if b == 'A1':  # NO + congruent
                    C = other(target, rng)
                    F = pick_not_in({C, target}, rng)
                    resp_correct = 'NO'
                    resp_cong = 'congruent'
    
                elif b == 'A2':  # NO + incongruent
                    C = other(target, rng)
                    F = C
                    resp_correct = 'NO'
                    resp_cong = 'incongruent'
    
                elif b == 'B1':  # YES + congruent
                    C = target
                    F = C
                    resp_correct = 'YES'
                    resp_cong = 'congruent'
    
                else:  # B2: YES + incongruent
                    C = target
                    F = other(target, rng)
                    resp_correct = 'YES'
                    resp_cong = 'incongruent'
    
                rows.append(dict(
                    n_back=0,
                    central_letter=C,
                    flanker_letter=F,
                    congruency=resp_cong,
                    is_target=int(resp_correct == 'YES'),
                    correctAns=resp_correct,
                    correctKey=('1' if resp_correct == 'YES' else '2'),
                    type=b,
                ))
    
        # randomize order with RNG
        rng.shuffle(rows)
        return rows
    
    
    # ======================================================
    # 1-BACK
    # ======================================================
    def make_1back(n_trials, rng, max_run=2):
        """
        make_1back():
            Builds a FULLY BALANCED 1-back block across the four trial types (A1, A2, B1, B2)
    
            Key features:
                • Balances type counts
                • Enforces all feasibility rules:
                    - A2 only possible if previous trial had F_prev != C_prev
                    - No more than 'max_run' repeats of same type in a row
                • Uses a seed previous state (C_prev, F_prev).
                • Uses backtracking with a hard call limit to avoid infinite loops.
    
            Returns:
                rows - list of dictionaries ready for PsychoPy trial loops.
        """
        # Set max recursive/backtracking calls based on n_trials
        max_calls = n_trials * 1e4
    
        # Seed "previous" state
        C_prev = rng.choice(letters)
        F_prev = rng.choice([letter for letter in letters if letter != C_prev])
    
        # Remaining quota for each type
        remaining = _type_counts(n_trials)  # dict: {'A1': n, 'A2': n, 'B1': n, 'B2': n}
        rows = []
    
        # Burner trial (t = 0): logged as first row but not assigned a type
        rows.append(dict(
            n_back=1,
            central_letter=C_prev,
            flanker_letter=F_prev,
            congruency=None,
            is_target=None,
            correctAns=None,
            correctKey=None,
            type=None,
        ))
    
        # --- Safety: limit total recursive/backtracking calls so we don't "freeze" ---
        call_count = {'n': 0}  # mutable wrapper so inner function can update
    
        # -----------------------------------------
        # Internal helper: which trial types are allowed next?
        # -----------------------------------------
        def candidates(Cp, Fp, last_type, run_len, remaining):
            """
            Determine which type choices are feasible for the next trial,
            based on:
                - remaining quota for each trial type
                - max_run constraint for the same trial type
                - A2 feasibility rule (needs F_prev != C_prev)
            """
            feas = []
            for b in ('A1', 'A2', 'B1', 'B2'):
                if remaining[b] == 0:
                    continue
                if b == last_type and run_len >= max_run:
                    continue
                if b == 'A2' and Fp == Cp:
                    # A2 (NO, hard) only possible when F_prev != C_prev
                    continue
                feas.append(b)
    
            if not feas:
                return []
    
            # Weighted random sampling without replacement:
            def priority(b):
                bias = 2.0 if b == 'A2' else 1.0
                return rng.random() ** (1.0 / (remaining[b] * bias))
    
            # Sort so that larger weights tend to come first
            ordered = sorted(feas, key=priority, reverse=True)
    
            return ordered
    
        # -----------------------------------------
        # Recursive backtracking scheduler
        # -----------------------------------------
        def step(Cp, Fp, last_type, run_len, remaining):
            """
            Recursively assign types and build the entire block.
            Backtracks whenever a choice leads to dead-end later.
            """
            # Bump global call counter and bail if too big
            call_count['n'] += 1
            if call_count['n'] > max_calls:
    
                # print out the types assigned so far for debugging
                assigned_types = [r['type'] for r in rows]
                print(f"C_prev={C_prev}, F_prev={F_prev} failed after {call_count['n']} calls."
                      f"Cp={Cp}, Fp={Fp}, remaining={remaining}. trial count: {len(assigned_types)}")
                raise RuntimeError("Exceeded max recursive calls in make_1back().")
    
            # Done when all type counts are zero
            if sum(remaining.values()) == 0:
                return True
    
            cand = candidates(Cp, Fp, last_type, run_len, remaining)
    
            # No valid type for next trial -> backtrack
            if not cand:
                return False
    
            for b in cand:
                # Generate next (C_t, F_t) based on type rules
                if b == 'B1':
                    # YES, easy: C_t = C_prev, F_t = C_t
                    C_t = Cp
                    F_t = C_t
                    resp_correct = 'YES'
                    resp_cong = 'congruent'
    
                elif b == 'B2':
                    # YES, hard: C_t = C_prev, F_t != C_t and != F_prev
                    C_t = Cp
                    F_t = pick_not_in({C_t, Fp}, rng)
                    resp_correct = 'YES'
                    resp_cong = 'incongruent'
    
                elif b == 'A1':
                    # NO, easy: C_t != C_prev, F_t != C_t, C_prev, F_prev
                    C_t = pick_not_in({Cp}, rng)
                    F_t = pick_not_in({C_t, Cp, Fp}, rng)
                    resp_correct = 'NO'
                    resp_cong = 'congruent'
    
                else:  # 'A2'
                    # NO, hard: C_t = F_prev (and F_prev != C_prev), F_t = C_t
                    C_t = Fp
                    F_t = C_t
                    resp_correct = 'NO'
                    resp_cong = 'incongruent'
    
                # Use up one slot of this type
                remaining[b] -= 1
    
                rows.append(dict(
                    n_back=1,
                    central_letter=C_t,
                    flanker_letter=F_t,
                    congruency=resp_cong,
                    is_target=int(resp_correct == 'YES'),
                    correctAns=resp_correct,
                    correctKey=('1' if resp_correct == 'YES' else '2'),
                    type=b,
                ))
    
                # Track run length for this type
                new_run = run_len + 1 if b == last_type else 1
    
                # Recurse; if success, return True
                if step(C_t, F_t, b, new_run, remaining):
                    return True
    
                # Backtrack
                rows.pop()
                remaining[b] += 1
    
            return False
    
        # -----------------------------------------
        # Constructing the 1-back block
        # -----------------------------------------
        step(C_prev, F_prev, last_type=None, run_len=0, remaining=remaining)
    
        return rows
    
    
    # ======================================================
    # PRACTICE + MAIN TRIAL SETUP
    # ======================================================
    practice0 = make_0back(n_trials=n_practice, target=target0, rng=rng)
    practice1 = make_1back(n_trials=n_practice, rng=rng)
    trials0 = make_0back(n_trials=n_main, target=target0, rng=rng)
    
    # Due to the complexity of balancing 1-back blocks, we regenerate until we get a valid one.
    while True:
        try:
            trials1 = make_1back(n_trials=n_main, rng=rng)
            break
        except RuntimeError:
            continue
    
    # ======================================================
    # Sanity checks
    # ======================================================
    # Create a histogram of A2 positions
    # import matplotlib.pyplot as plt
    # A2_positions = [i for i, r in enumerate(trials1) if r['type'] == 'A2']
    # plt.hist(A2_positions, bins=10, range=(0, len(trials1)))
    # plt.show()
    
    # Make sure type counts are balanced across trials
    for name, lst in [
        ('practice0', practice0),
        ('practice1', practice1),
        ('0back',     trials0),
        ('1back',     trials1),
    ]:
        counts = {'A1': 0, 'A2': 0, 'B1': 0, 'B2': 0}
        for r in lst:
            assert 'congruency' in r and 'correctKey' in r and 'type' in r, \
                f"Missing labels in {name}"
            if r['type'] is not None:
                counts[r['type']] += 1
    
        base = len(lst) // 4
        for b in counts.keys():
            assert counts[b] == base, f"Unbalanced type counts in {name}: {counts}"
    
    
    # --- Initialize components for Routine "welcome" ---
    welcomeText = visual.TextStim(win=win, name='welcomeText',
        text='Welcome!\n\n\nIn this task, you will see a CENTER letter.\nLetters also appear on the sides, but you can ignore them.\n\n\nPress any of the white keys to continue',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    welcomeKey = keyboard.Keyboard(deviceName='welcomeKey')
    
    # --- Initialize components for Routine "practiceIntro" ---
    pracInstruct = visual.TextStim(win=win, name='pracInstruct',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    pracInstructKey = keyboard.Keyboard(deviceName='pracInstructKey')
    
    # --- Initialize components for Routine "ITI" ---
    fixation = visual.TextStim(win=win, name='fixation',
        text='+',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.2, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "trial" ---
    leftFlank = visual.TextStim(win=win, name='leftFlank',
        text='',
        font='Arial',
        units='norm', pos=(-0.15, 0), draggable=False, height=0.15, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    central = visual.TextStim(win=win, name='central',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.15, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    rightFlank = visual.TextStim(win=win, name='rightFlank',
        text='',
        font='Arial',
        units='norm', pos=(0.15, 0), draggable=False, height=0.15, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    respCue = visual.TextStim(win=win, name='respCue',
        text='< >',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.2, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    earlyResp = keyboard.Keyboard(deviceName='earlyResp')
    resp = keyboard.Keyboard(deviceName='resp')
    
    # --- Initialize components for Routine "feedback" ---
    fbText = visual.TextStim(win=win, name='fbText',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "ITI" ---
    fixation = visual.TextStim(win=win, name='fixation',
        text='+',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.2, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "trialIntro" ---
    instruct = visual.TextStim(win=win, name='instruct',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    instructKey = keyboard.Keyboard(deviceName='instructKey')
    
    # --- Initialize components for Routine "ITI" ---
    fixation = visual.TextStim(win=win, name='fixation',
        text='+',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.2, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "trial" ---
    leftFlank = visual.TextStim(win=win, name='leftFlank',
        text='',
        font='Arial',
        units='norm', pos=(-0.15, 0), draggable=False, height=0.15, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    central = visual.TextStim(win=win, name='central',
        text='',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.15, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    rightFlank = visual.TextStim(win=win, name='rightFlank',
        text='',
        font='Arial',
        units='norm', pos=(0.15, 0), draggable=False, height=0.15, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    respCue = visual.TextStim(win=win, name='respCue',
        text='< >',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.2, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    earlyResp = keyboard.Keyboard(deviceName='earlyResp')
    resp = keyboard.Keyboard(deviceName='resp')
    
    # --- Initialize components for Routine "ITI" ---
    fixation = visual.TextStim(win=win, name='fixation',
        text='+',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.2, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "__end__" ---
    text_thank_you = visual.TextStim(win=win, name='text_thank_you',
        text='Thank you. You have completed this task!',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    read_thank_you = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_thank_you',    name='read_thank_you'
    )
    read_thank_you.setVolume(1.0)
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "__start__" ---
    # create an object to store info about Routine __start__
    __start__ = data.Routine(
        name='__start__',
        components=[],
    )
    __start__.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for __start__
    __start__.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    __start__.tStart = globalClock.getTime(format='float')
    __start__.status = STARTED
    __start__.maxDuration = None
    # keep track of which components have finished
    __start__Components = __start__.components
    for thisComponent in __start__.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "__start__" ---
    __start__.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            __start__.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in __start__.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "__start__" ---
    for thisComponent in __start__.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for __start__
    __start__.tStop = globalClock.getTime(format='float')
    __start__.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "__start__" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    nbackBlocks = data.TrialHandler2(
        name='nbackBlocks',
        nReps=len(n_back_blocks), 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(nbackBlocks)  # add the loop to the experiment
    thisNbackBlock = nbackBlocks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisNbackBlock.rgb)
    if thisNbackBlock != None:
        for paramName in thisNbackBlock:
            globals()[paramName] = thisNbackBlock[paramName]
    
    for thisNbackBlock in nbackBlocks:
        currentLoop = nbackBlocks
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisNbackBlock.rgb)
        if thisNbackBlock != None:
            for paramName in thisNbackBlock:
                globals()[paramName] = thisNbackBlock[paramName]
        
        # --- Prepare to start Routine "welcome" ---
        # create an object to store info about Routine welcome
        welcome = data.Routine(
            name='welcome',
            components=[welcomeText, welcomeKey],
        )
        welcome.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from blockSetup
        # --- Determine n-back level for this block ---
        # n_back_blocks is defined in the __start__ routine: [0, 1]
        n_back = n_back_blocks[nbackBlocks.thisRepN]  # 0-back for first block, 1-back for second
        
        # --- Select the correct condition lists for this block ---
        if n_back == 0:
            practiceBlockTrials  = practice0   # 0-back practice
            blockTrials          = trials0     # main 0-back trials
        
        else:
            practiceBlockTrials  = practice1   # 1-back practice
            blockTrials          = trials1     # main 1-back trials
        
        # --- Show welcome screen only for FIRST block ---
        if nbackBlocks.thisRepN > 0:
            continueRoutine = False
        
            # End the main experiment trial block from the previous loop
            dev.activate_line(bitmask=block_end_code)
            core.wait(0.5)  # wait 500ms before next block start trigger
        
        # create starting attributes for welcomeKey
        welcomeKey.keys = []
        welcomeKey.rt = []
        _welcomeKey_allKeys = []
        # store start times for welcome
        welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        welcome.tStart = globalClock.getTime(format='float')
        welcome.status = STARTED
        welcome.maxDuration = None
        # keep track of which components have finished
        welcomeComponents = welcome.components
        for thisComponent in welcome.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "welcome" ---
        # if trial has changed, end Routine now
        if isinstance(nbackBlocks, data.TrialHandler2) and thisNbackBlock.thisN != nbackBlocks.thisTrial.thisN:
            continueRoutine = False
        welcome.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *welcomeText* updates
            
            # if welcomeText is starting this frame...
            if welcomeText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                welcomeText.frameNStart = frameN  # exact frame index
                welcomeText.tStart = t  # local t and not account for scr refresh
                welcomeText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(welcomeText, 'tStartRefresh')  # time at next scr refresh
                # update status
                welcomeText.status = STARTED
                welcomeText.setAutoDraw(True)
            
            # if welcomeText is active this frame...
            if welcomeText.status == STARTED:
                # update params
                pass
            
            # *welcomeKey* updates
            waitOnFlip = False
            
            # if welcomeKey is starting this frame...
            if welcomeKey.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                welcomeKey.frameNStart = frameN  # exact frame index
                welcomeKey.tStart = t  # local t and not account for scr refresh
                welcomeKey.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(welcomeKey, 'tStartRefresh')  # time at next scr refresh
                # update status
                welcomeKey.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(welcomeKey.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(welcomeKey.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if welcomeKey.status == STARTED and not waitOnFlip:
                theseKeys = welcomeKey.getKeys(keyList=['3', '4', '5', '6'], ignoreKeys=["escape"], waitRelease=True)
                _welcomeKey_allKeys.extend(theseKeys)
                if len(_welcomeKey_allKeys):
                    welcomeKey.keys = _welcomeKey_allKeys[-1].name  # just the last key pressed
                    welcomeKey.rt = _welcomeKey_allKeys[-1].rt
                    welcomeKey.duration = _welcomeKey_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                welcome.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in welcome.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "welcome" ---
        for thisComponent in welcome.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for welcome
        welcome.tStop = globalClock.getTime(format='float')
        welcome.tStopRefresh = tThisFlipGlobal
        # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "practiceIntro" ---
        # create an object to store info about Routine practiceIntro
        practiceIntro = data.Routine(
            name='practiceIntro',
            components=[pracInstruct, pracInstructKey],
        )
        practiceIntro.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from setPracInstruct
        phase = 'practice'
        
        if n_back == 0:
            practice_instruct_msg = (
                "0-Back Practice\n"
                "\n"
                "\n"
                "Rule: Look for the target letter " + target0 + "\n"
                "\n"
                "Press GREEN key if the CENTER letter is " + target0 + ".\n"
                "Press RED key for any other letter.\n"
                "\n"
                "Wait for the '< >' prompt before responding.\n"
                "\n"
                "\n"
                "Press the green key to start practice trials"
            )
        
        else:
            practice_instruct_msg = (
                "1-Back Practice\n"
                "\n"
                "\n"
                "Rule: Compare the current CENTER letter to the PREVIOUS CENTER letter\n"
                "\n"
                "Press GREEN key if the CENTER letter is the SAME as the previous CENTER letter.\n"
                "Press RED key if the CENTER letter is DIFFERENT from the previous CENTER letter.\n"
                "\n"
                "Wait for the '< >' prompt before responding.\n"
                "\n"
                "\n"
                "Press the green key to start practice trials"
            )
        
        # Start a practice block
        dev.activate_line(bitmask=block_start_code)
        # no need to wait 500ms as instructions are displayed
        
        pracInstruct.setText(practice_instruct_msg)
        # create starting attributes for pracInstructKey
        pracInstructKey.keys = []
        pracInstructKey.rt = []
        _pracInstructKey_allKeys = []
        # store start times for practiceIntro
        practiceIntro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practiceIntro.tStart = globalClock.getTime(format='float')
        practiceIntro.status = STARTED
        practiceIntro.maxDuration = None
        # keep track of which components have finished
        practiceIntroComponents = practiceIntro.components
        for thisComponent in practiceIntro.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practiceIntro" ---
        # if trial has changed, end Routine now
        if isinstance(nbackBlocks, data.TrialHandler2) and thisNbackBlock.thisN != nbackBlocks.thisTrial.thisN:
            continueRoutine = False
        practiceIntro.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *pracInstruct* updates
            
            # if pracInstruct is starting this frame...
            if pracInstruct.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                pracInstruct.frameNStart = frameN  # exact frame index
                pracInstruct.tStart = t  # local t and not account for scr refresh
                pracInstruct.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pracInstruct, 'tStartRefresh')  # time at next scr refresh
                # update status
                pracInstruct.status = STARTED
                pracInstruct.setAutoDraw(True)
            
            # if pracInstruct is active this frame...
            if pracInstruct.status == STARTED:
                # update params
                pass
            
            # *pracInstructKey* updates
            waitOnFlip = False
            
            # if pracInstructKey is starting this frame...
            if pracInstructKey.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                pracInstructKey.frameNStart = frameN  # exact frame index
                pracInstructKey.tStart = t  # local t and not account for scr refresh
                pracInstructKey.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pracInstructKey, 'tStartRefresh')  # time at next scr refresh
                # update status
                pracInstructKey.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(pracInstructKey.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(pracInstructKey.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if pracInstructKey.status == STARTED and not waitOnFlip:
                theseKeys = pracInstructKey.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=True)
                _pracInstructKey_allKeys.extend(theseKeys)
                if len(_pracInstructKey_allKeys):
                    pracInstructKey.keys = _pracInstructKey_allKeys[-1].name  # just the last key pressed
                    pracInstructKey.rt = _pracInstructKey_allKeys[-1].rt
                    pracInstructKey.duration = _pracInstructKey_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                practiceIntro.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in practiceIntro.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practiceIntro" ---
        for thisComponent in practiceIntro.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practiceIntro
        practiceIntro.tStop = globalClock.getTime(format='float')
        practiceIntro.tStopRefresh = tThisFlipGlobal
        # the Routine "practiceIntro" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "ITI" ---
        # create an object to store info about Routine ITI
        ITI = data.Routine(
            name='ITI',
            components=[fixation],
        )
        ITI.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from trigger_iti
        iti_trigger_started = False
        
        # store start times for ITI
        ITI.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        ITI.tStart = globalClock.getTime(format='float')
        ITI.status = STARTED
        ITI.maxDuration = None
        # keep track of which components have finished
        ITIComponents = ITI.components
        for thisComponent in ITI.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "ITI" ---
        # if trial has changed, end Routine now
        if isinstance(nbackBlocks, data.TrialHandler2) and thisNbackBlock.thisN != nbackBlocks.thisTrial.thisN:
            continueRoutine = False
        ITI.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation.frameNStop = frameN  # exact frame index
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            # Run 'Each Frame' code from trigger_iti
            if fixation.status == STARTED and not iti_trigger_started:
                win.callOnFlip(dev.activate_line, bitmask=iti_start_code)
                iti_trigger_started = True
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                ITI.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ITI.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "ITI" ---
        for thisComponent in ITI.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for ITI
        ITI.tStop = globalClock.getTime(format='float')
        ITI.tStopRefresh = tThisFlipGlobal
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if ITI.maxDurationReached:
            routineTimer.addTime(-ITI.maxDuration)
        elif ITI.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # set up handler to look after randomisation of conditions etc
        practice = data.TrialHandler2(
            name='practice',
            nReps=len(practiceBlockTrials), 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(practice)  # add the loop to the experiment
        thisPractice = practice.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPractice.rgb)
        if thisPractice != None:
            for paramName in thisPractice:
                globals()[paramName] = thisPractice[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisPractice in practice:
            currentLoop = practice
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisPractice.rgb)
            if thisPractice != None:
                for paramName in thisPractice:
                    globals()[paramName] = thisPractice[paramName]
            
            # --- Prepare to start Routine "trial" ---
            # create an object to store info about Routine trial
            trial = data.Routine(
                name='trial',
                components=[leftFlank, central, rightFlank, respCue, earlyResp, resp],
            )
            trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from trialSetup
            # The following variables are already defined at this point:
            #   phase == 'practice'  (set in practiceIntro)  OR
            #   phase == 'main'      (set in trialIntro)
            #
            # blockSetup in the welcome routine created:
            #   practiceBlockTrials : list of practice rows for this block
            #   blockTrials         : list of main rows for this block
            
            # Get that row from the pre-built list
            if phase == 'practice':
                row = practiceBlockTrials[practice.thisRepN]
            
            else:
                row = blockTrials[trials.thisRepN]
            
            # Jittered cueOnset: time when '+' appears, AFTER the letters disappear
            cueOnset = rng.uniform(respCueJitter[0], respCueJitter[1]) + stimulusDur  # stimulusDur = 0.5
            
            leftFlank.setText(row['flanker_letter']*3)
            central.setText(row['central_letter'])
            rightFlank.setText(row['flanker_letter']*3)
            # create starting attributes for earlyResp
            earlyResp.keys = []
            earlyResp.rt = []
            _earlyResp_allKeys = []
            # create starting attributes for resp
            resp.keys = []
            resp.rt = []
            _resp_allKeys = []
            # Run 'Begin Routine' code from trigger_trial
            letter_trigger_started = False
            cue_trigger_started = False
            
            # store start times for trial
            trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial.tStart = globalClock.getTime(format='float')
            trial.status = STARTED
            thisExp.addData('trial.started', trial.tStart)
            trial.maxDuration = None
            # keep track of which components have finished
            trialComponents = trial.components
            for thisComponent in trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            # if trial has changed, end Routine now
            if isinstance(practice, data.TrialHandler2) and thisPractice.thisN != practice.thisTrial.thisN:
                continueRoutine = False
            trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *leftFlank* updates
                
                # if leftFlank is starting this frame...
                if leftFlank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    leftFlank.frameNStart = frameN  # exact frame index
                    leftFlank.tStart = t  # local t and not account for scr refresh
                    leftFlank.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(leftFlank, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'leftFlank.started')
                    # update status
                    leftFlank.status = STARTED
                    leftFlank.setAutoDraw(True)
                
                # if leftFlank is active this frame...
                if leftFlank.status == STARTED:
                    # update params
                    pass
                
                # if leftFlank is stopping this frame...
                if leftFlank.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > leftFlank.tStartRefresh + stimulusDur-frameTolerance:
                        # keep track of stop time/frame for later
                        leftFlank.tStop = t  # not accounting for scr refresh
                        leftFlank.tStopRefresh = tThisFlipGlobal  # on global time
                        leftFlank.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'leftFlank.stopped')
                        # update status
                        leftFlank.status = FINISHED
                        leftFlank.setAutoDraw(False)
                
                # *central* updates
                
                # if central is starting this frame...
                if central.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    central.frameNStart = frameN  # exact frame index
                    central.tStart = t  # local t and not account for scr refresh
                    central.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(central, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'central.started')
                    # update status
                    central.status = STARTED
                    central.setAutoDraw(True)
                
                # if central is active this frame...
                if central.status == STARTED:
                    # update params
                    pass
                
                # if central is stopping this frame...
                if central.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > central.tStartRefresh + stimulusDur-frameTolerance:
                        # keep track of stop time/frame for later
                        central.tStop = t  # not accounting for scr refresh
                        central.tStopRefresh = tThisFlipGlobal  # on global time
                        central.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'central.stopped')
                        # update status
                        central.status = FINISHED
                        central.setAutoDraw(False)
                
                # *rightFlank* updates
                
                # if rightFlank is starting this frame...
                if rightFlank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    rightFlank.frameNStart = frameN  # exact frame index
                    rightFlank.tStart = t  # local t and not account for scr refresh
                    rightFlank.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(rightFlank, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rightFlank.started')
                    # update status
                    rightFlank.status = STARTED
                    rightFlank.setAutoDraw(True)
                
                # if rightFlank is active this frame...
                if rightFlank.status == STARTED:
                    # update params
                    pass
                
                # if rightFlank is stopping this frame...
                if rightFlank.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > rightFlank.tStartRefresh + stimulusDur-frameTolerance:
                        # keep track of stop time/frame for later
                        rightFlank.tStop = t  # not accounting for scr refresh
                        rightFlank.tStopRefresh = tThisFlipGlobal  # on global time
                        rightFlank.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'rightFlank.stopped')
                        # update status
                        rightFlank.status = FINISHED
                        rightFlank.setAutoDraw(False)
                
                # *respCue* updates
                
                # if respCue is starting this frame...
                if respCue.status == NOT_STARTED and tThisFlip >= cueOnset-frameTolerance:
                    # keep track of start time/frame for later
                    respCue.frameNStart = frameN  # exact frame index
                    respCue.tStart = t  # local t and not account for scr refresh
                    respCue.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(respCue, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'respCue.started')
                    # update status
                    respCue.status = STARTED
                    respCue.setAutoDraw(True)
                
                # if respCue is active this frame...
                if respCue.status == STARTED:
                    # update params
                    pass
                
                # if respCue is stopping this frame...
                if respCue.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > respCue.tStartRefresh + cueDur-frameTolerance:
                        # keep track of stop time/frame for later
                        respCue.tStop = t  # not accounting for scr refresh
                        respCue.tStopRefresh = tThisFlipGlobal  # on global time
                        respCue.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'respCue.stopped')
                        # update status
                        respCue.status = FINISHED
                        respCue.setAutoDraw(False)
                
                # *earlyResp* updates
                waitOnFlip = False
                
                # if earlyResp is starting this frame...
                if earlyResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    earlyResp.frameNStart = frameN  # exact frame index
                    earlyResp.tStart = t  # local t and not account for scr refresh
                    earlyResp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(earlyResp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'earlyResp.started')
                    # update status
                    earlyResp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(earlyResp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(earlyResp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if earlyResp is stopping this frame...
                if earlyResp.status == STARTED:
                    # is it time to stop? (based on local clock)
                    if tThisFlip > cueOnset-frameTolerance:
                        # keep track of stop time/frame for later
                        earlyResp.tStop = t  # not accounting for scr refresh
                        earlyResp.tStopRefresh = tThisFlipGlobal  # on global time
                        earlyResp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'earlyResp.stopped')
                        # update status
                        earlyResp.status = FINISHED
                        earlyResp.status = FINISHED
                if earlyResp.status == STARTED and not waitOnFlip:
                    theseKeys = earlyResp.getKeys(keyList=['1', '2'], ignoreKeys=["escape"], waitRelease=False)
                    _earlyResp_allKeys.extend(theseKeys)
                    if len(_earlyResp_allKeys):
                        earlyResp.keys = [key.name for key in _earlyResp_allKeys]  # storing all keys
                        earlyResp.rt = [key.rt for key in _earlyResp_allKeys]
                        earlyResp.duration = [key.duration for key in _earlyResp_allKeys]
                
                # *resp* updates
                waitOnFlip = False
                
                # if resp is starting this frame...
                if resp.status == NOT_STARTED and tThisFlip >= cueOnset-frameTolerance:
                    # keep track of start time/frame for later
                    resp.frameNStart = frameN  # exact frame index
                    resp.tStart = t  # local t and not account for scr refresh
                    resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'resp.started')
                    # update status
                    resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if resp is stopping this frame...
                if resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > resp.tStartRefresh + cueDur-frameTolerance:
                        # keep track of stop time/frame for later
                        resp.tStop = t  # not accounting for scr refresh
                        resp.tStopRefresh = tThisFlipGlobal  # on global time
                        resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'resp.stopped')
                        # update status
                        resp.status = FINISHED
                        resp.status = FINISHED
                if resp.status == STARTED and not waitOnFlip:
                    theseKeys = resp.getKeys(keyList=['1','2'], ignoreKeys=["escape"], waitRelease=False)
                    _resp_allKeys.extend(theseKeys)
                    if len(_resp_allKeys):
                        resp.keys = _resp_allKeys[0].name  # just the first key pressed
                        resp.rt = _resp_allKeys[0].rt
                        resp.duration = _resp_allKeys[0].duration
                        # a response ends the routine
                        continueRoutine = False
                # Run 'Each Frame' code from trigger_trial
                if rightFlank.status == STARTED and not letter_trigger_started:
                    win.callOnFlip(dev.activate_line, bitmask=letter_start_code)
                    letter_trigger_started = True
                
                if respCue.status == STARTED and not cue_trigger_started:
                    win.callOnFlip(dev.activate_line, bitmask=cue_start_code)
                    cue_trigger_started = True
                
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial
            trial.tStop = globalClock.getTime(format='float')
            trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial.stopped', trial.tStop)
            # Run 'End Routine' code from trialSetup
            thisExp.addData('central_letter', row['central_letter'])
            thisExp.addData('flanker_letter', row['flanker_letter'])
            thisExp.addData('congruency', row['congruency'])
            thisExp.addData('correctKey', row['correctKey'])
            
            # check responses
            if earlyResp.keys in ['', [], None]:  # No response was made
                earlyResp.keys = None
            practice.addData('earlyResp.keys',earlyResp.keys)
            if earlyResp.keys != None:  # we had a response
                practice.addData('earlyResp.rt', earlyResp.rt)
                practice.addData('earlyResp.duration', earlyResp.duration)
            # check responses
            if resp.keys in ['', [], None]:  # No response was made
                resp.keys = None
            practice.addData('resp.keys',resp.keys)
            if resp.keys != None:  # we had a response
                practice.addData('resp.rt', resp.rt)
                practice.addData('resp.duration', resp.duration)
            # the Routine "trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "feedback" ---
            # create an object to store info about Routine feedback
            feedback = data.Routine(
                name='feedback',
                components=[fbText],
            )
            feedback.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from setFbText
            # ---- Feedback logic for practice ----
            # Get early press (before cue)
            early_key = earlyResp.keys
            
            # Get main press (after cue)
            main_key = resp.keys
            
            # 1) Too fast: pressed BEFORE cue onset
            if early_key is not None:
                fb_msg = "Too fast. Wait for the cue to respond."
            
            # 2) Otherwise, if they responded after cue, score it
            elif main_key is not None:
                fb_msg = "Correct!" if main_key == row['correctKey'] else "Wrong"
            
            # 3) No response at all
            else:
                fb_msg = "Too slow. Please try to respond quickly."
            
            fbText.setText(fb_msg)
            # Run 'Begin Routine' code from trigger_fb
            feedback_trigger_started = False
            
            # store start times for feedback
            feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            feedback.tStart = globalClock.getTime(format='float')
            feedback.status = STARTED
            feedback.maxDuration = None
            # keep track of which components have finished
            feedbackComponents = feedback.components
            for thisComponent in feedback.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "feedback" ---
            # if trial has changed, end Routine now
            if isinstance(practice, data.TrialHandler2) and thisPractice.thisN != practice.thisTrial.thisN:
                continueRoutine = False
            feedback.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 3.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *fbText* updates
                
                # if fbText is starting this frame...
                if fbText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fbText.frameNStart = frameN  # exact frame index
                    fbText.tStart = t  # local t and not account for scr refresh
                    fbText.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fbText, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    fbText.status = STARTED
                    fbText.setAutoDraw(True)
                
                # if fbText is active this frame...
                if fbText.status == STARTED:
                    # update params
                    pass
                
                # if fbText is stopping this frame...
                if fbText.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fbText.tStartRefresh + 3-frameTolerance:
                        # keep track of stop time/frame for later
                        fbText.tStop = t  # not accounting for scr refresh
                        fbText.tStopRefresh = tThisFlipGlobal  # on global time
                        fbText.frameNStop = frameN  # exact frame index
                        # update status
                        fbText.status = FINISHED
                        fbText.setAutoDraw(False)
                # Run 'Each Frame' code from trigger_fb
                if fbText.status == STARTED and not feedback_trigger_started:
                    win.callOnFlip(dev.activate_line, bitmask=feedback_start_code)
                    feedback_trigger_started = True
                
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    feedback.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in feedback.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "feedback" ---
            for thisComponent in feedback.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for feedback
            feedback.tStop = globalClock.getTime(format='float')
            feedback.tStopRefresh = tThisFlipGlobal
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if feedback.maxDurationReached:
                routineTimer.addTime(-feedback.maxDuration)
            elif feedback.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-3.000000)
            
            # --- Prepare to start Routine "ITI" ---
            # create an object to store info about Routine ITI
            ITI = data.Routine(
                name='ITI',
                components=[fixation],
            )
            ITI.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from trigger_iti
            iti_trigger_started = False
            
            # store start times for ITI
            ITI.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            ITI.tStart = globalClock.getTime(format='float')
            ITI.status = STARTED
            ITI.maxDuration = None
            # keep track of which components have finished
            ITIComponents = ITI.components
            for thisComponent in ITI.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "ITI" ---
            # if trial has changed, end Routine now
            if isinstance(practice, data.TrialHandler2) and thisPractice.thisN != practice.thisTrial.thisN:
                continueRoutine = False
            ITI.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *fixation* updates
                
                # if fixation is starting this frame...
                if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fixation.frameNStart = frameN  # exact frame index
                    fixation.tStart = t  # local t and not account for scr refresh
                    fixation.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    fixation.status = STARTED
                    fixation.setAutoDraw(True)
                
                # if fixation is active this frame...
                if fixation.status == STARTED:
                    # update params
                    pass
                
                # if fixation is stopping this frame...
                if fixation.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fixation.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        fixation.tStop = t  # not accounting for scr refresh
                        fixation.tStopRefresh = tThisFlipGlobal  # on global time
                        fixation.frameNStop = frameN  # exact frame index
                        # update status
                        fixation.status = FINISHED
                        fixation.setAutoDraw(False)
                # Run 'Each Frame' code from trigger_iti
                if fixation.status == STARTED and not iti_trigger_started:
                    win.callOnFlip(dev.activate_line, bitmask=iti_start_code)
                    iti_trigger_started = True
                
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    ITI.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in ITI.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "ITI" ---
            for thisComponent in ITI.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for ITI
            ITI.tStop = globalClock.getTime(format='float')
            ITI.tStopRefresh = tThisFlipGlobal
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if ITI.maxDurationReached:
                routineTimer.addTime(-ITI.maxDuration)
            elif ITI.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            thisExp.nextEntry()
            
        # completed len(practiceBlockTrials) repeats of 'practice'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "trialIntro" ---
        # create an object to store info about Routine trialIntro
        trialIntro = data.Routine(
            name='trialIntro',
            components=[instruct, instructKey],
        )
        trialIntro.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from setInstruct
        phase = 'main'
        
        if n_back == 0:
            instruct_msg = (
                "0-Back Trials\n"
                "\n"
                "\n"
                "Target letter: " + target0 + "\n"
                "\n"
                "Press GREEN key if the CENTER letter is " + target0 + ".\n"
                "Press RED key for any other letter.\n"
                "\n"
                "You can ignore the letters on the sides, and you won't receive feedback.\n"
                "Wait for the '< >' prompt before responding.\n"
                "\n"
                "\n"
                "Press the green key to begin"
            )
        
        else:
            instruct_msg = (
                "1-Back Trials\n"
                "\n"
                "\n"
                "Press GREEN key if the CENTER letter is the SAME as the previous CENTER letter.\n"
                "Press RED key if the CENTER letter is DIFFERENT from the previous CENTER letter.\n"
                "\n"
                "You can ignore the letters on the sides, and you won't receive feedback.\n"
                "Wait for the '< >' prompt before responding.\n"
                "\n"
                "\n"
                "Press the green key to begin"
            )
        
        # End the practice block
        dev.activate_line(bitmask=block_end_code)
        core.wait(0.5)  # wait 500ms before next block start trigger
        
        # Start a main experiment trial block
        dev.activate_line(bitmask=block_start_code)
        # no need to wait 500ms as instructions are displayed
        
        instruct.setText(instruct_msg)
        # create starting attributes for instructKey
        instructKey.keys = []
        instructKey.rt = []
        _instructKey_allKeys = []
        # store start times for trialIntro
        trialIntro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trialIntro.tStart = globalClock.getTime(format='float')
        trialIntro.status = STARTED
        trialIntro.maxDuration = None
        # keep track of which components have finished
        trialIntroComponents = trialIntro.components
        for thisComponent in trialIntro.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trialIntro" ---
        # if trial has changed, end Routine now
        if isinstance(nbackBlocks, data.TrialHandler2) and thisNbackBlock.thisN != nbackBlocks.thisTrial.thisN:
            continueRoutine = False
        trialIntro.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *instruct* updates
            
            # if instruct is starting this frame...
            if instruct.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instruct.frameNStart = frameN  # exact frame index
                instruct.tStart = t  # local t and not account for scr refresh
                instruct.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instruct, 'tStartRefresh')  # time at next scr refresh
                # update status
                instruct.status = STARTED
                instruct.setAutoDraw(True)
            
            # if instruct is active this frame...
            if instruct.status == STARTED:
                # update params
                pass
            
            # *instructKey* updates
            waitOnFlip = False
            
            # if instructKey is starting this frame...
            if instructKey.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instructKey.frameNStart = frameN  # exact frame index
                instructKey.tStart = t  # local t and not account for scr refresh
                instructKey.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instructKey, 'tStartRefresh')  # time at next scr refresh
                # update status
                instructKey.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(instructKey.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(instructKey.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if instructKey.status == STARTED and not waitOnFlip:
                theseKeys = instructKey.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=True)
                _instructKey_allKeys.extend(theseKeys)
                if len(_instructKey_allKeys):
                    instructKey.keys = _instructKey_allKeys[-1].name  # just the last key pressed
                    instructKey.rt = _instructKey_allKeys[-1].rt
                    instructKey.duration = _instructKey_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trialIntro.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialIntro.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trialIntro" ---
        for thisComponent in trialIntro.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trialIntro
        trialIntro.tStop = globalClock.getTime(format='float')
        trialIntro.tStopRefresh = tThisFlipGlobal
        # the Routine "trialIntro" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "ITI" ---
        # create an object to store info about Routine ITI
        ITI = data.Routine(
            name='ITI',
            components=[fixation],
        )
        ITI.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from trigger_iti
        iti_trigger_started = False
        
        # store start times for ITI
        ITI.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        ITI.tStart = globalClock.getTime(format='float')
        ITI.status = STARTED
        ITI.maxDuration = None
        # keep track of which components have finished
        ITIComponents = ITI.components
        for thisComponent in ITI.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "ITI" ---
        # if trial has changed, end Routine now
        if isinstance(nbackBlocks, data.TrialHandler2) and thisNbackBlock.thisN != nbackBlocks.thisTrial.thisN:
            continueRoutine = False
        ITI.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation.frameNStop = frameN  # exact frame index
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            # Run 'Each Frame' code from trigger_iti
            if fixation.status == STARTED and not iti_trigger_started:
                win.callOnFlip(dev.activate_line, bitmask=iti_start_code)
                iti_trigger_started = True
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                ITI.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ITI.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "ITI" ---
        for thisComponent in ITI.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for ITI
        ITI.tStop = globalClock.getTime(format='float')
        ITI.tStopRefresh = tThisFlipGlobal
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if ITI.maxDurationReached:
            routineTimer.addTime(-ITI.maxDuration)
        elif ITI.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler2(
            name='trials',
            nReps=len(blockTrials), 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial in trials:
            currentLoop = trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "trial" ---
            # create an object to store info about Routine trial
            trial = data.Routine(
                name='trial',
                components=[leftFlank, central, rightFlank, respCue, earlyResp, resp],
            )
            trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from trialSetup
            # The following variables are already defined at this point:
            #   phase == 'practice'  (set in practiceIntro)  OR
            #   phase == 'main'      (set in trialIntro)
            #
            # blockSetup in the welcome routine created:
            #   practiceBlockTrials : list of practice rows for this block
            #   blockTrials         : list of main rows for this block
            
            # Get that row from the pre-built list
            if phase == 'practice':
                row = practiceBlockTrials[practice.thisRepN]
            
            else:
                row = blockTrials[trials.thisRepN]
            
            # Jittered cueOnset: time when '+' appears, AFTER the letters disappear
            cueOnset = rng.uniform(respCueJitter[0], respCueJitter[1]) + stimulusDur  # stimulusDur = 0.5
            
            leftFlank.setText(row['flanker_letter']*3)
            central.setText(row['central_letter'])
            rightFlank.setText(row['flanker_letter']*3)
            # create starting attributes for earlyResp
            earlyResp.keys = []
            earlyResp.rt = []
            _earlyResp_allKeys = []
            # create starting attributes for resp
            resp.keys = []
            resp.rt = []
            _resp_allKeys = []
            # Run 'Begin Routine' code from trigger_trial
            letter_trigger_started = False
            cue_trigger_started = False
            
            # store start times for trial
            trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial.tStart = globalClock.getTime(format='float')
            trial.status = STARTED
            thisExp.addData('trial.started', trial.tStart)
            trial.maxDuration = None
            # keep track of which components have finished
            trialComponents = trial.components
            for thisComponent in trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *leftFlank* updates
                
                # if leftFlank is starting this frame...
                if leftFlank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    leftFlank.frameNStart = frameN  # exact frame index
                    leftFlank.tStart = t  # local t and not account for scr refresh
                    leftFlank.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(leftFlank, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'leftFlank.started')
                    # update status
                    leftFlank.status = STARTED
                    leftFlank.setAutoDraw(True)
                
                # if leftFlank is active this frame...
                if leftFlank.status == STARTED:
                    # update params
                    pass
                
                # if leftFlank is stopping this frame...
                if leftFlank.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > leftFlank.tStartRefresh + stimulusDur-frameTolerance:
                        # keep track of stop time/frame for later
                        leftFlank.tStop = t  # not accounting for scr refresh
                        leftFlank.tStopRefresh = tThisFlipGlobal  # on global time
                        leftFlank.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'leftFlank.stopped')
                        # update status
                        leftFlank.status = FINISHED
                        leftFlank.setAutoDraw(False)
                
                # *central* updates
                
                # if central is starting this frame...
                if central.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    central.frameNStart = frameN  # exact frame index
                    central.tStart = t  # local t and not account for scr refresh
                    central.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(central, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'central.started')
                    # update status
                    central.status = STARTED
                    central.setAutoDraw(True)
                
                # if central is active this frame...
                if central.status == STARTED:
                    # update params
                    pass
                
                # if central is stopping this frame...
                if central.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > central.tStartRefresh + stimulusDur-frameTolerance:
                        # keep track of stop time/frame for later
                        central.tStop = t  # not accounting for scr refresh
                        central.tStopRefresh = tThisFlipGlobal  # on global time
                        central.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'central.stopped')
                        # update status
                        central.status = FINISHED
                        central.setAutoDraw(False)
                
                # *rightFlank* updates
                
                # if rightFlank is starting this frame...
                if rightFlank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    rightFlank.frameNStart = frameN  # exact frame index
                    rightFlank.tStart = t  # local t and not account for scr refresh
                    rightFlank.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(rightFlank, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rightFlank.started')
                    # update status
                    rightFlank.status = STARTED
                    rightFlank.setAutoDraw(True)
                
                # if rightFlank is active this frame...
                if rightFlank.status == STARTED:
                    # update params
                    pass
                
                # if rightFlank is stopping this frame...
                if rightFlank.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > rightFlank.tStartRefresh + stimulusDur-frameTolerance:
                        # keep track of stop time/frame for later
                        rightFlank.tStop = t  # not accounting for scr refresh
                        rightFlank.tStopRefresh = tThisFlipGlobal  # on global time
                        rightFlank.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'rightFlank.stopped')
                        # update status
                        rightFlank.status = FINISHED
                        rightFlank.setAutoDraw(False)
                
                # *respCue* updates
                
                # if respCue is starting this frame...
                if respCue.status == NOT_STARTED and tThisFlip >= cueOnset-frameTolerance:
                    # keep track of start time/frame for later
                    respCue.frameNStart = frameN  # exact frame index
                    respCue.tStart = t  # local t and not account for scr refresh
                    respCue.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(respCue, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'respCue.started')
                    # update status
                    respCue.status = STARTED
                    respCue.setAutoDraw(True)
                
                # if respCue is active this frame...
                if respCue.status == STARTED:
                    # update params
                    pass
                
                # if respCue is stopping this frame...
                if respCue.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > respCue.tStartRefresh + cueDur-frameTolerance:
                        # keep track of stop time/frame for later
                        respCue.tStop = t  # not accounting for scr refresh
                        respCue.tStopRefresh = tThisFlipGlobal  # on global time
                        respCue.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'respCue.stopped')
                        # update status
                        respCue.status = FINISHED
                        respCue.setAutoDraw(False)
                
                # *earlyResp* updates
                waitOnFlip = False
                
                # if earlyResp is starting this frame...
                if earlyResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    earlyResp.frameNStart = frameN  # exact frame index
                    earlyResp.tStart = t  # local t and not account for scr refresh
                    earlyResp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(earlyResp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'earlyResp.started')
                    # update status
                    earlyResp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(earlyResp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(earlyResp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if earlyResp is stopping this frame...
                if earlyResp.status == STARTED:
                    # is it time to stop? (based on local clock)
                    if tThisFlip > cueOnset-frameTolerance:
                        # keep track of stop time/frame for later
                        earlyResp.tStop = t  # not accounting for scr refresh
                        earlyResp.tStopRefresh = tThisFlipGlobal  # on global time
                        earlyResp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'earlyResp.stopped')
                        # update status
                        earlyResp.status = FINISHED
                        earlyResp.status = FINISHED
                if earlyResp.status == STARTED and not waitOnFlip:
                    theseKeys = earlyResp.getKeys(keyList=['1', '2'], ignoreKeys=["escape"], waitRelease=False)
                    _earlyResp_allKeys.extend(theseKeys)
                    if len(_earlyResp_allKeys):
                        earlyResp.keys = [key.name for key in _earlyResp_allKeys]  # storing all keys
                        earlyResp.rt = [key.rt for key in _earlyResp_allKeys]
                        earlyResp.duration = [key.duration for key in _earlyResp_allKeys]
                
                # *resp* updates
                waitOnFlip = False
                
                # if resp is starting this frame...
                if resp.status == NOT_STARTED and tThisFlip >= cueOnset-frameTolerance:
                    # keep track of start time/frame for later
                    resp.frameNStart = frameN  # exact frame index
                    resp.tStart = t  # local t and not account for scr refresh
                    resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'resp.started')
                    # update status
                    resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if resp is stopping this frame...
                if resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > resp.tStartRefresh + cueDur-frameTolerance:
                        # keep track of stop time/frame for later
                        resp.tStop = t  # not accounting for scr refresh
                        resp.tStopRefresh = tThisFlipGlobal  # on global time
                        resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'resp.stopped')
                        # update status
                        resp.status = FINISHED
                        resp.status = FINISHED
                if resp.status == STARTED and not waitOnFlip:
                    theseKeys = resp.getKeys(keyList=['1','2'], ignoreKeys=["escape"], waitRelease=False)
                    _resp_allKeys.extend(theseKeys)
                    if len(_resp_allKeys):
                        resp.keys = _resp_allKeys[0].name  # just the first key pressed
                        resp.rt = _resp_allKeys[0].rt
                        resp.duration = _resp_allKeys[0].duration
                        # a response ends the routine
                        continueRoutine = False
                # Run 'Each Frame' code from trigger_trial
                if rightFlank.status == STARTED and not letter_trigger_started:
                    win.callOnFlip(dev.activate_line, bitmask=letter_start_code)
                    letter_trigger_started = True
                
                if respCue.status == STARTED and not cue_trigger_started:
                    win.callOnFlip(dev.activate_line, bitmask=cue_start_code)
                    cue_trigger_started = True
                
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial
            trial.tStop = globalClock.getTime(format='float')
            trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial.stopped', trial.tStop)
            # Run 'End Routine' code from trialSetup
            thisExp.addData('central_letter', row['central_letter'])
            thisExp.addData('flanker_letter', row['flanker_letter'])
            thisExp.addData('congruency', row['congruency'])
            thisExp.addData('correctKey', row['correctKey'])
            
            # check responses
            if earlyResp.keys in ['', [], None]:  # No response was made
                earlyResp.keys = None
            trials.addData('earlyResp.keys',earlyResp.keys)
            if earlyResp.keys != None:  # we had a response
                trials.addData('earlyResp.rt', earlyResp.rt)
                trials.addData('earlyResp.duration', earlyResp.duration)
            # check responses
            if resp.keys in ['', [], None]:  # No response was made
                resp.keys = None
            trials.addData('resp.keys',resp.keys)
            if resp.keys != None:  # we had a response
                trials.addData('resp.rt', resp.rt)
                trials.addData('resp.duration', resp.duration)
            # the Routine "trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "ITI" ---
            # create an object to store info about Routine ITI
            ITI = data.Routine(
                name='ITI',
                components=[fixation],
            )
            ITI.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from trigger_iti
            iti_trigger_started = False
            
            # store start times for ITI
            ITI.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            ITI.tStart = globalClock.getTime(format='float')
            ITI.status = STARTED
            ITI.maxDuration = None
            # keep track of which components have finished
            ITIComponents = ITI.components
            for thisComponent in ITI.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "ITI" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            ITI.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *fixation* updates
                
                # if fixation is starting this frame...
                if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fixation.frameNStart = frameN  # exact frame index
                    fixation.tStart = t  # local t and not account for scr refresh
                    fixation.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    fixation.status = STARTED
                    fixation.setAutoDraw(True)
                
                # if fixation is active this frame...
                if fixation.status == STARTED:
                    # update params
                    pass
                
                # if fixation is stopping this frame...
                if fixation.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fixation.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        fixation.tStop = t  # not accounting for scr refresh
                        fixation.tStopRefresh = tThisFlipGlobal  # on global time
                        fixation.frameNStop = frameN  # exact frame index
                        # update status
                        fixation.status = FINISHED
                        fixation.setAutoDraw(False)
                # Run 'Each Frame' code from trigger_iti
                if fixation.status == STARTED and not iti_trigger_started:
                    win.callOnFlip(dev.activate_line, bitmask=iti_start_code)
                    iti_trigger_started = True
                
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    ITI.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in ITI.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "ITI" ---
            for thisComponent in ITI.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for ITI
            ITI.tStop = globalClock.getTime(format='float')
            ITI.tStopRefresh = tThisFlipGlobal
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if ITI.maxDurationReached:
                routineTimer.addTime(-ITI.maxDuration)
            elif ITI.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            thisExp.nextEntry()
            
        # completed len(blockTrials) repeats of 'trials'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed len(n_back_blocks) repeats of 'nbackBlocks'
    
    
    # --- Prepare to start Routine "__end__" ---
    # create an object to store info about Routine __end__
    __end__ = data.Routine(
        name='__end__',
        components=[text_thank_you, read_thank_you],
    )
    __end__.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    read_thank_you.setSound('resource/thank_you.wav', secs=2.7, hamming=True)
    read_thank_you.setVolume(1.0, log=False)
    read_thank_you.seek(0)
    # Run 'Begin Routine' code from trigger_trial_block_end
    # End of main experiment trial block
    dev.activate_line(bitmask=block_end_code)
    # no need to wait 500ms as this routine lasts 3.0s before experiment ends
    
    # store start times for __end__
    __end__.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    __end__.tStart = globalClock.getTime(format='float')
    __end__.status = STARTED
    thisExp.addData('__end__.started', __end__.tStart)
    __end__.maxDuration = None
    # keep track of which components have finished
    __end__Components = __end__.components
    for thisComponent in __end__.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "__end__" ---
    __end__.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_thank_you* updates
        
        # if text_thank_you is starting this frame...
        if text_thank_you.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_thank_you.frameNStart = frameN  # exact frame index
            text_thank_you.tStart = t  # local t and not account for scr refresh
            text_thank_you.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_thank_you, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_thank_you.status = STARTED
            text_thank_you.setAutoDraw(True)
        
        # if text_thank_you is active this frame...
        if text_thank_you.status == STARTED:
            # update params
            pass
        
        # if text_thank_you is stopping this frame...
        if text_thank_you.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_thank_you.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                text_thank_you.tStop = t  # not accounting for scr refresh
                text_thank_you.tStopRefresh = tThisFlipGlobal  # on global time
                text_thank_you.frameNStop = frameN  # exact frame index
                # update status
                text_thank_you.status = FINISHED
                text_thank_you.setAutoDraw(False)
        
        # *read_thank_you* updates
        
        # if read_thank_you is starting this frame...
        if read_thank_you.status == NOT_STARTED and t >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            read_thank_you.frameNStart = frameN  # exact frame index
            read_thank_you.tStart = t  # local t and not account for scr refresh
            read_thank_you.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_thank_you.status = STARTED
            read_thank_you.play()  # start the sound (it finishes automatically)
        
        # if read_thank_you is stopping this frame...
        if read_thank_you.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > read_thank_you.tStartRefresh + 2.7-frameTolerance or read_thank_you.isFinished:
                # keep track of stop time/frame for later
                read_thank_you.tStop = t  # not accounting for scr refresh
                read_thank_you.tStopRefresh = tThisFlipGlobal  # on global time
                read_thank_you.frameNStop = frameN  # exact frame index
                # update status
                read_thank_you.status = FINISHED
                read_thank_you.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[read_thank_you]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            __end__.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in __end__.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "__end__" ---
    for thisComponent in __end__.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for __end__
    __end__.tStop = globalClock.getTime(format='float')
    __end__.tStopRefresh = tThisFlipGlobal
    thisExp.addData('__end__.stopped', __end__.tStop)
    read_thank_you.pause()  # ensure sound has stopped at end of Routine
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if __end__.maxDurationReached:
        routineTimer.addTime(-__end__.maxDuration)
    elif __end__.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    # Run 'End Experiment' code from eeg
    # Stop EEG recording
    dev.activate_line(bitmask=127)  # trigger 127 will stop EEG
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
