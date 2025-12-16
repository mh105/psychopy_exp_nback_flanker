# N-back flanker task (no pupil)
Last edit: 12/16/2025

## Edit history
- 12/16/2025 by Alex He - increased ITI fixation cross size and duration from 1s to 2s
- 11/28/2025 by Alex He - created finalized first draft version by Harjinder Singh

## Description
This is a working memory and cognitive control task, combining the Eriksen Flanker task with the N-back task. This behavioral paradigm is adapted from the following reference paper:

Scharinger, C., Soutschek, A., Schubert, T., & Gerjets, P. (2015). When flanker meets the n‐back: What EEG and pupil dilation data reveal about the interplay between the two central‐executive working memory functions inhibition and updating. Psychophysiology, 52(10), 1293-1304.

In each trial, a center letter and 3x flanking letters on each side appear on the screen for 500ms. Participants are instructed to without their response wait until a response cue '< >' show up on the screen. The response cue will appear at an interval uniformly distributed between 1.5 and 3.0 seconds after the end of the letter stimuli, and it stays on for up to 3.0 seconds. The trial ends as soon as a response is made after the response cue has appeared. If response(s) is made before the onset of the response cue, the trial silently logs the extra responses but do not end the trial, which is not treated as failed withholding. After the trial ends, a 1 second inter-trial interval is displayed with a center fixation cross to catch evolving electrophysiology and to provide a separation between trials.

This task is designed for Dr. Pegah Kahali to study electrophysiology of Parkinson patients during a complex cognitive control task with motor responses as well as motor withholding patterns. The combination of N-back and Flanker tasks allows one to investigate different components of working memory functions, i.e., inhibition and updating, in these patients, as well as in response to deep-brain stimulation (DBS) treatment. Both 0-back and 1-back versions are included here in the design to provide a within-subject control to study working memory specific processes.

## Outcome measures
- Behavioral performance as hit-rate in 0-back and 1-back conditions
- Behavioral congruency effect
- Oscillation properties at the onset of letter stimulus
- Oscillation dynamics until and around the response cue
- Electrophysiological signal contrasts between congruent and incongruent trials
- Electrophysiological signal contrasts between successfully and failed withholding trials
