function AdaptationRun(varargin)
close all;

%% Adaptation Run Experiment code:
%%   Using Audapter to read in audio input, track formants, and formant shift audio feedback to the user.
%%
%%   The first part of the code is the basic configuration settings of the program (which are not editable by a user [only editable through code]).
%%
%%   After this, the program has user modifiable settings, which can be edited the through a GUI (see the USE_GUI variable), this can be used to edit parameters for
%%     the experiment. Primarily parameters relating to the behavior of Audapter (which controls formant tracking and the behavior of how formants will be shifted by Audapter)
%%
%%   After this, we decide on the flow of the experiment using the simulus data and perturbation index data (stim_data.txt and idx_data.txt), which will control the order
%%     in which words are presented to the subject (which the subject is supposed to speak), and how much, if any formant shifting will be applied to the audio feedback in their headphones.
%%     a perturbation index of 0 means that the sound that was recorded will be output to the test subjects headphones directly, whereas a perturbation index of 10 mean that the audio
%%     will have its formants shifted as much as possible (indexes 1 to 9 represent amounts of formant shifting between no shifting and maximum shifting, e.g. 5 is 50% of the maximum, 1 is 10% of the maximum, etc.)
%%
%%   Based on the information read from the stimulus / perturbation files as well as the parameters set in the gui, we run the experiment flow. If the user enabled the "Run Screening" option,
%%     we will ignore the information in stim_data.txt and idx_data.txt and instead use a preset experimental run (predefined in the code, see SCREENING_STIM_DATA and NUM_WORD_ITERATIONS, for the behavior of the "screening mode"),
%%     and the screening will not use and perturbation (so perturbation index is always 0 when "Run Screening" is enabled). SCREENING_STIM_DATA defines the words for the screening and NUM_WORD_ITERATIONS defines how many
%%     times each word is repeated.
%%   The experiment flow will show a word for a short period of time as well as take in audio (through Audapter) for a short period of time (for each word). Output data is recorded including the data for tracking formants,
%%     the recorded audio data itself, the formant shifted audio data, labels for the time intervals of the vowel data, and lots of other information about the sample (there are wav, txt, mat, FBW, and LBL files for each sample, which all have different information)
%%
%%   After each stimulus / word has run (and its corresponding output data has been saved), we run some cumulative statistical calculations (based on all the trials) which are saved to a csv file.
%%   There is also some code for visualization (graphing) of the tracked / shifted formants, although I THINK THIS PART OF THE CODE MAY BE OUTDATED and it may only show the tracking for the very last trial of the experiment
%%   (I would recommend investigating this further if you want to be sure of this)
%%
%%   NOTE 1: I've marked some important parts of the code with @IMPORTANT, be sure to look at these to make sure you understand the basic structure of the code.
%%     One of these marks the location where a trial finishes and we finally grab the information of that trial from Audapter to see the tracked formant data.
%%     This is a very important part of the code if you are interested in utilizing the formant data from each trial.
%%
%%   NOTE 2: For Audapter, it's very valuable to look at the Audapter manual for information about the output data and the input parameters (this is one of the most vital things to understand for this code):
%%     The latest Audapter manual can be found here: https://sites.bu.edu/guentherlab/files/2022/09/AudapterManual_2.1.5.pdf (Although it's probably a good idea to check the version of Audapter used in this experiment because it may not be the latest Audapter version)
%%
%%   -pberry 1/12/2026

%%
%% Configurations - Variables to adjust the behavior of the program:
%%

% Audio driver to use:
AUDIO_INTERFACE_NAME = 'ASIO4ALL v2';
% Which display monitor to use. This only matters for multi-display devices. Use 1 for the first display, 2 for the second display, etc.
SCREEN_TO_USE = 2;
% Audio sampling rate in hertz (before downsampling):
SAMPLING_RATE = 48000;
% Downsampling factor, the higher this is, the less computational load is required for audio processing (ideally for reducing latency at the expense of accuracy). The default for Audapter is 3.
DOWNSAMPLING_FACTOR = 3;
% The number of samples of audio (before downsampling) that is required before outputting the first chunk of audio:
FRAME_LENGTH = 64;

% A setting that lets the program know that we only expect one single vowel to be spoken in a test run, this allows us to more easily identify the single vowel we're interested in looking at (this should be false in experiments where multiple vowels are recorded in a single test run):
SINGLE_VOWEL_MODE = true;
% Enable the GUI which allows for editing various parameters / options for the experiment:
USE_GUI = true;
% Whether to use a window or use fullscreen for the experiment:
WINDOWED_MODE = false;

% --- Adaptive agent integration ---
USE_ADAPTIVE_AGENT = true;          % perturbation each trial comes from the Python agent
ADAPTIVE_WORD      = 'hate';      % change the word here
ADAPTIVE_NUM_TRIALS = 30;
ADAPTIVE_NUM_BASELINE =10;            % opening zero-shift trials, averaged into the agent's starting anchor (must be < ADAPTIVE_NUM_TRIALS)
AGENT_BRIDGE_DIR   = 'C:\Users\mitti\Documents\AdaptationExperiment';  % folder holding agent_bridge.py

% 'who''d' /u/
%TARGET_U_F1_MALE   = 349;   TARGET_U_F2_MALE   = 962;    
%TARGET_U_F1_FEMALE = 429;   TARGET_U_F2_FEMALE = 1104;

% "heat" /i/ 
% TARGET_U_F1_MALE   = 342;  TARGET_U_F2_MALE   = 2322;
% TARGET_U_F1_FEMALE = 437;  TARGET_U_F2_FEMALE = 2761;

% "hat" /a/ 
TARGET_U_F1_MALE   = 562;  TARGET_U_F2_MALE   = 1401;
TARGET_U_F1_FEMALE = 736;  TARGET_U_F2_FEMALE = 1702;


% Placeholder; overwritten by the sex selector right after the GUI closes.
TARGET_U_F1 = TARGET_U_F1_FEMALE;   TARGET_U_F2 = TARGET_U_F2_FEMALE;

PERT_HZ_TO_AMP = 1.0;                % CALIBRATION: pertAmp units per Hz of intended shift (verify, see below)

% --- Bad-take rejection: a detected vowel is only USED if its measured
% formants are physically plausible. Garbage takes (silence, noise, tracker
% glitches like F1=89/F2=386) are treated exactly like a no-vowel trial:
% skipped, estimate kept, pairing reset, baseline anchor uncontaminated.
VOWEL_F1_MIN = 200;  VOWEL_F1_MAX = 1500;   % Hz, generous bounds for /u/-region vowels
VOWEL_F2_MIN = 700;  VOWEL_F2_MAX = 3000;   % Hz

DEFAULT_GENDER = 'female';
STIM_DATA_FILENAME = 'stim_data.txt'; % This text file has the stimulus words for the experiment
IDX_DATA_FILENAME = 'idx_data.txt';   % This text file has the perturbation amounts for each stimulus word for the experiment
MAX_NUM_RUNS = 0; % Anything 0 or below means any number of runs
FIRST_SCREENING_WORD = 'heed'; % This word is shown to the participant in the screening, but the output formant is not recorded
LAST_SCREENING_WORD = 'hayd';  % This word is shown to the participant in the screening, but the output formant is not recorded
% These are the simulus words we will use when we are running the "screening mode":
SCREENING_STIM_DATA = {'heed', 'hid', 'who''d', 'hood', 'hod', 'had', 'hoe''d', 'head', 'hayd', 'hawed'};
% @NOTE: The screening vowel names contains some values representing unicode
% characters. For example, 03B5 is the hex value of the unicode character
% for epsilon, so to represent this in unicode, we use \x03B5.
%
% We use these special greek letters to make the output data more readable to speech researchers (to follow the convention of speech vowel labels)
% 028A: small epsilon
% 0251: small alpha
% 00E6: small ae
% 03B5: greek small epsilon
% 0254: latin small open o (backwards c)
SCREENING_VOWEL_NAMES = {'/i/', '/I/', '/u/', '/\x028A/', '/\x0251/', '/\x00E6/', '/o/', '/\x03B5/', '/e/', '/\x0254/'};
NUM_UNIQUE_STIM_WORDS = size(SCREENING_STIM_DATA, 2);
VOWEL_CALCULATION_INDICES = [3, 9; 9, 5; 5, 3]; % /u/ to /e/, /e/ to /a/, /a/ to /u/
PERCENT_FORMANT_CHANGE = [0.0394, -0.0881; -0.0165, -0.0244; 0.0189, 0.0312;];
NUM_VOWEL_CALCULATIONS = size(VOWEL_CALCULATION_INDICES, 1);
NUM_WORD_ITERATIONS = 3; % How many times a word will be repeated (Number of Screening Runs), this later becomes 1 if it's not a screening

if MAX_NUM_RUNS > 0
    NUM_UNIQUE_STIM_WORDS = min(NUM_UNIQUE_STIM_WORDS, MAX_NUM_RUNS);
    NUM_VOWEL_CALCULATIONS = 0;
end

% Covert vowel names to unicode representations:
for iter = 1 : NUM_UNIQUE_STIM_WORDS SCREENING_VOWEL_NAMES{1, iter} = sprintf(SCREENING_VOWEL_NAMES{1, iter}); end

pert_phi_values = zeros(NUM_VOWEL_CALCULATIONS, 1);
euclidean_dist_values = zeros(NUM_VOWEL_CALCULATIONS, 1);
pert_amp_values = zeros(NUM_VOWEL_CALCULATIONS, 1);
run_screening = true;

WORD_DISPLAY_TIME = 2; % Should be 2
AUDIO_INTAKE_TIME = 1; % Should Be 1
DEBUG_OVERRIDE_SUBJECT_ID_ENABLED = false;
DEBUG_OVERRIDE_SUBJECT_ID = 'subject0_7';

if ~isempty(varargin) gender = varargin{1};
else gender = DEFAULT_GENDER; end
gender_is_female = false;
if (~strcmp(gender, 'male')) gender_is_female = true; end

row_names = {'Subject ID' 'Female'          'F1 Min' 'F1 Max' 'F2 Min' 'F2 Max' 'LBk' 'LBb' 'Perturbation Amplitude' 'Perturbation Phi' 'RMS Threshold' 'Run Screening' 'Num Screening Runs'};
vars_data = {'subject0';  gender_is_female; 0;       5000;    0;       5000;    0;    0;    0.4;                     0.75;              0.050;          run_screening;  NUM_WORD_ITERATIONS };
should_start = false;

% Function below is used by the gui to signal that the experiment should start once the gui is closed:
    function start_callback(~, ~, ~)
        should_start = true;
    end
% GUI for adjusting settings for the experiment before starting:
if USE_GUI
    gdy = 19.1*(size(vars_data, 1) + 1);

    table_figure = figure('Position', [100 250 420 (60 + gdy)], 'name', 'Variables');
    gui_table = uitable('Parent', table_figure, 'Position', [25 50 370 ...
        gdy], 'Data', vars_data, 'ColumnWidth', {125}, ...
        'RowName', row_names, 'ColumnName', {'Value'});
    gui_table.ColumnEditable(1) = true;

    uicontrol('Style', 'pushbutton', 'String', 'Start Program', ...
        'Parent', table_figure, 'Position', [100 10 200 30], 'Callback', @start_callback); % Start Button

    while (~should_start)
        pause(0.2);
        if (isempty(findobj('type', 'figure', 'name', 'Variables'))) return; end % User closed the window
    end

    vars_data = gui_table.Data;
end

gender_is_female = vars_data{2};
if (gender_is_female) gender = 'female';
else gender = 'male';
end

% --- Pick the sex-matched /u/ target from the Female checkbox ---
if gender_is_female
    TARGET_U_F1 = TARGET_U_F1_FEMALE;   TARGET_U_F2 = TARGET_U_F2_FEMALE;
else
    TARGET_U_F1 = TARGET_U_F1_MALE;     TARGET_U_F2 = TARGET_U_F2_MALE;
end
fprintf('[target] %s -> /u/ target (%.0f, %.0f) Hz\n', gender, TARGET_U_F1, TARGET_U_F2);

run_screening = vars_data{12};
NUM_WORD_ITERATIONS = vars_data{13};
subject_id = vars_data{1};

close all;

% Function to read a file and split it into an array line strings:
    function file_lines = get_file_lines(filename)
        all_file_lines = splitlines(fileread(filename));
        file_lines = {};
        num_file_lines = 0;
        for iter = 1 : size(all_file_lines, 1)
            str = strtrim(all_file_lines{iter, 1});
            if 0 ~= strcmp(str, '') continue; end
            num_file_lines = num_file_lines + 1;
            file_lines(num_file_lines, 1) = {str};
        end
    end

% Stimulus data is read through a data file (These are the words to use):
stim_data = get_file_lines(STIM_DATA_FILENAME);
num_stim_words = size(stim_data, 1);

indices_file_lines = get_file_lines(IDX_DATA_FILENAME);
num_indices = size(indices_file_lines, 1);
% Index data is read through a data file (These are how much to perturb each word):
idx_data = zeros(num_indices, 1);
% Convert the indices in the index file to numbers we can reference in an array:
for iter = 1 : num_indices idx_data(iter, 1) = str2num(indices_file_lines{iter, 1}); end

num_stim_words = min(num_stim_words, num_indices);
if MAX_NUM_RUNS > 0 num_stim_words = min(num_stim_words, MAX_NUM_RUNS); end
% If we're running the screening we aren't referencing the stim data file and are instead showing the vowels from the screening stim data array:
if run_screening num_stim_words = NUM_UNIQUE_STIM_WORDS*NUM_WORD_ITERATIONS; end
num_indices = num_stim_words;

% --- Adaptive override: placed AFTER file reads/clamps so they cannot clobber it ---
if USE_ADAPTIVE_AGENT
    run_screening  = false;
    stim_data      = repmat({ADAPTIVE_WORD}, ADAPTIVE_NUM_TRIALS, 1);
    num_stim_words = ADAPTIVE_NUM_TRIALS;
    idx_data       = zeros(num_stim_words, 1);
    num_indices    = num_stim_words;
end

if ~run_screening
    NUM_UNIQUE_STIM_WORDS = num_stim_words;
    NUM_WORD_ITERATIONS = 1;
    SCREENING_VOWEL_NAMES = cell(num_stim_words, 1);
    for iter = 1 : num_stim_words SCREENING_VOWEL_NAMES(1, iter) = {'-'}; end
end

% Initialize matrices to hold experiment output info filenames based on each piece of stimulus data (e.g. wav audio files, matrices, FBW files, label (LBL) files, and text files):
avg_mid_fmt_data = zeros(num_stim_words, 2);
avg_mid_sfmt_data = zeros(num_stim_words, 2);
stim_ids = zeros(num_stim_words, 1);
out_names = cell(num_stim_words, 2);
out_namesData = cell(num_stim_words, 1);
out_namesFormData = cell(num_stim_words, 1);
out_namesFBW = cell(num_stim_words, 2);
out_namesLBL = cell(num_stim_words, 2);
for stim_id = 1 : num_stim_words
    out_names(stim_id, 1) = {sprintf('%02d.wav', stim_id)};
    out_names(stim_id, 2) = {sprintf('%02dR.wav', stim_id)};
    out_namesData(stim_id, 1) = {sprintf('%02d.mat', stim_id)};
    out_namesFBW(stim_id, 1) = {sprintf('%02d.FBW', stim_id)};
    out_namesFBW(stim_id, 2) = {sprintf('%02dR.FBW', stim_id)};
    out_namesLBL(stim_id, 1) = {sprintf('%02d.LBL', stim_id)};
    out_namesLBL(stim_id, 2) = {sprintf('%02dR.LBL', stim_id)};
    out_namesFormData(stim_id, 1) = {sprintf('%02d.txt', stim_id)};
    stim_ids(stim_id, 1) = stim_id;
end

if run_screening
    stim_data = cell(num_stim_words, 1);
    for iter = 1 : num_stim_words
        test_data_index = int32(mod(iter - 1, NUM_UNIQUE_STIM_WORDS)) + 1;
        stim_data(iter, 1) = SCREENING_STIM_DATA(test_data_index);
    end
end

if 7 ~= exist('subject_data', 'dir') mkdir('subject_data'); end;
cd('subject_data');
if ~DEBUG_OVERRIDE_SUBJECT_ID_ENABLED
    orig_subject_id = subject_id;
    subject_id = sprintf('%s_1', orig_subject_id);
    iter = 2;
    while 7 == exist(subject_id, 'dir')
        subject_id = sprintf('%s_%d', orig_subject_id, iter);
        iter = iter + 1;
    end
else
    subject_id = DEBUG_OVERRIDE_SUBJECT_ID;
    warning('off', 'MATLAB:MKDIR:DirectoryExists');
end
mkdir(subject_id);
cd(subject_id);

%% Visualization configuration
GRAY = [0.5, 0.5, 0.5];
OST_MULT = 250;
LEGEND_FONT_SIZE = 8;

WAV_NOISE_FILENAME = 'mtbabble48k.wav';

%%
Audapter('deviceName', AUDIO_INTERFACE_NAME);
% Audapter('setParam', 'downFact', DOWNSAMPLING_FACTOR, 0);
% Audapter('setParam', 'sRate', SAMPLING_RATE/DOWNSAMPLING_FACTOR, 0);
% Audapter('setParam', 'frameLen', FRAME_LENGTH/DOWNSAMPLING_FACTOR, 0);

B_VIS = 0;
B_VIS_FORMATS = 0;
B_VIS_OST = 0;
VIS_NAME = '';

Audapter('ost', '', 0);
Audapter('pcf', '', 0);

params = getAudapterDefaultParams(gender);

% Inputting: Audapter settings based on information from the gui:

params.F1Min = vars_data{3};
params.F1Max = vars_data{4};
params.F2Min = vars_data{5};
params.F2Max = vars_data{6};

params.LBk = vars_data{7};
params.LBb = vars_data{8};
params.pertF2 = linspace(0, 5000, 257);
pert_amp = vars_data{9} * ones(1, 257);
pert_phi = vars_data{10} * pi * ones(1, 257);
params.pertAmp = pert_amp;
params.pertPhi = pert_phi;
params.bTrack = 1;
params.bShift = 1;
params.bRatioShift = 1;
params.bMelShift = 0;
params.bDetect = 1;
params.rmsThresh = vars_data{11}; % Default is somewhere around 0.011 %@IMPORTANT: Mess with this until the formant tracking looks correct!
% Also mess with rmsRatio

% Read in the noise wav file and feed it into Audapter, this will end up playing through Audapter in the output audio:
max_pb_size = Audapter('getMaxPBLen');
filename = sprintf('../../%s', WAV_NOISE_FILENAME);
check_file(filename);
[w, fs] = read_audio(filename);

% We need to adjust the audio sample data for the noise to fit the settings Audapter expects, so we
%   adjust the samples based on downsampling and sampling rate, then limit based on the maximum
%   playback size max_pb_size (from Audapter('getMaxPBLen')):
if fs ~= params.sr * params.downFact
    w = resample(w, params.sr * params.downFact, fs);
end
if length(w) > max_pb_size
    w = w(1 : max_pb_size);
end
Audapter('setParam', 'datapb', w, 1);

% Select feedback mode for Audapter:
if isempty(fsic(varargin, 'fb'))
    params.fb = 1;
else
    fb_mode = varargin{fsic(varargin, 'fb') + 1};
    if ~(fb_mode >= 0 && fb_mode <= 4 && floor(fb_mode) == fb_mode)
        error('Invalid fb mode: %d', fb_mode);
    end

    if fb_mode == 3
        params.fb3Gain = 0.1;
    end

    fprintf(1, 'Setting fb to %d\n', fb_mode);
    params.fb = varargin{fsic(varargin, 'fb') + 1};
end

params.trialLen = 2;
params.rampLen = 0.2;
%% Initialize PsychToolbox (Used for displaying text on the screen)
KbQueueCreate(0);
KbQueueStart(0);
WaitSecs(3);
Screen('Preference', 'SuppressAllWarnings', 1);
Screen('Preference', 'VisualDebugLevel', 1); % Disables welcome screen
Screen('Preference', 'SkipSyncTests', 1);
if WINDOWED_MODE Screen('Preference', 'WindowShieldingLevel', 500); end
screen_to_use = 0;
list_of_displays = Screen('Screens', 0);
if size(list_of_displays, 2) > 1 screen_to_use = SCREEN_TO_USE; end
if WINDOWED_MODE w = Screen('OpenWindow', screen_to_use, 0, [200, 200, 1000, 650]);
else w = Screen('OpenWindow', screen_to_use); end

% PsychToolbox settings: Which font to use and the background color:
white = WhiteIndex(w); % pixel value for white
black = BlackIndex(w); % pixel value for black
if ~WINDOWED_MODE HideCursor; end
Screen('TextFont',w, 'Arial');
Screen('TextSize',w, 100);
Screen('FillRect', w, black);
Screen(w, 'Flip');
WaitSecs(3);
%%

if USE_ADAPTIVE_AGENT
    % Run  pyenv('Version','...venv...python.exe')  ONCE per MATLAB session before
    % the first AdaptationRun -- you can't change the Python version once it's loaded.
    addpath(AGENT_BRIDGE_DIR);   % so plot_adaptation_diagnostics.m is found regardless of cd
    if count(py.sys.path, AGENT_BRIDGE_DIR) == 0
        insert(py.sys.path, int32(0), AGENT_BRIDGE_DIR);
    end
    py.importlib.import_module('agent_bridge');
    py.agent_bridge.make_agent(TARGET_U_F1, TARGET_U_F2, int32(0));
    py.agent_bridge.reset_session();
    log = struct('trial',[], 'obs',zeros(0,2), 'act',zeros(0,2), ...
        'shift',zeros(0,2), 'k',zeros(0,2), 'cross',zeros(0,2), 'base',[]);
    next_pert_amp = zeros(1, 257);   % baseline trials = no perturbation
    next_pert_phi = zeros(1, 257);
    baseline_sum   = [0.0, 0.0];     % running sum of mid-vowel (F1,F2) over baseline trials
    baseline_count = 0;              % how many valid baseline observations we have
    agent_primed   = false;          % becomes true once the averaged anchor is handed to the agent
end

% --- Per-trial logging for post-run visualization ---
log_obs    = nan(ADAPTIVE_NUM_TRIALS, 2);   % measured mid-vowel (F1,F2)
log_sobs   = nan(ADAPTIVE_NUM_TRIALS, 2);   % shifted mid-vowel (what they heard)
log_action = nan(ADAPTIVE_NUM_TRIALS, 2);   % agent action (dF1,dF2) Hz, computed this trial
log_estk   = nan(ADAPTIVE_NUM_TRIALS, 2);   % estimated gains (k_F1,k_F2)
log_cross  = nan(ADAPTIVE_NUM_TRIALS, 2);   % estimated cross terms
log_dist   = nan(ADAPTIVE_NUM_TRIALS, 1);   % distance of measured prod to target
log_phase  = zeros(ADAPTIVE_NUM_TRIALS, 1); % 0=skipped, 1=baseline, 2=control

while (true)
    num_words_to_use = num_stim_words;
    start_stim_id = 1;
    if run_screening
        num_words_to_use = NUM_WORD_ITERATIONS*NUM_UNIQUE_STIM_WORDS + 1;
        start_stim_id = 0;
    end
    %
    % @IMPORTANT: This is the core loop for showing each individual word (NOTE: The experiements can be set up with words that repeat):
    %   recording audio while tracking and shifting formants (Audapter) then taking the recorded / tracked data and outputting information to matrices and files afterward:
    %
    for stim_id = start_stim_id : num_words_to_use
        % Checking the keyboard if the escape key was pressed / if we should quit:
        [event, num_events_left] = KbEventGet(0);
        while (true)
            if isempty(event)
                break;
            else
                if event.CookedKey == 27 && event.Pressed == 1 % Escape Key Was Pressed, we should exit
                    % Exit:
                    KbQueueStop(0);
                    KbQueueRelease(0);
                    Screen('CloseAll');
                    save('AdaptationRun', 'idx_data', 'out_names', 'out_namesData', 'out_namesFormData', ...
                        'stim_data', 'stim_id');
                    cd('../..');
                    return;
                end
            end
            if num_events_left <= 0 break;
            else [event, num_events_left] = KbEventGet(0); end
        end

        test_data_index = int32(mod(stim_id - 1, NUM_UNIQUE_STIM_WORDS)) + 1;
        fprintf('run: %d\n', stim_id);

        % Initializing Audapter based on the parameters we had set up earlier in the code (see the code that sets the values in 'params'):
        AudapterIO('init', params);
        Audapter('reset');
        WaitSecs(1);

        % @IMPORTANT: Choose which word to use for this run of the test (based on whether this is a screening or not):
        if run_screening
            if stim_id == 0                    word = FIRST_SCREENING_WORD;
            elseif stim_id == num_words_to_use word = LAST_SCREENING_WORD;
            else                               word = SCREENING_STIM_DATA{test_data_index};
            end
        else
            word = stim_data{stim_id};
        end

        % Draw the word in Psych Toolbox:
        [nx, ny, textbounds] = DrawFormattedText(w, word, 'center', 'center', white);

        % Get the perturbation amount from the index file, otherwise have no perturbation if this is just a screening:
        if run_screening pert_idx = 0;
        else pert_idx = idx_data(stim_id);
        end

        if USE_ADAPTIVE_AGENT
            params.fb     = 1;                 % speech only
            params.bRatioShift = 0;            % agent acts in Hz -> apply pertAmp as absolute Hz, not a ratio
            params.pertAmp = next_pert_amp;     % set by the agent after the previous trial
            params.pertPhi = next_pert_phi;
        else
            % ----- original index-driven logic, unchanged -----
            if (pert_idx == 0) params.pertPhi = zeros(size(params.pertPhi));
            else               params.pertPhi = pert_phi; end

    		%
    		% Determine how much we will perturb the formants, how much they will be shifted, based on the perturbation index and the amplitude set in the gui:
    		%
    		% Note on the meaning of params.fb (From the Audapter manual):
    		% Feedback mode.
    		% 0: mute (play no sound)
    		% 1: normal (speech only)
    		% 2: noise only
    		% 3: speech + noise. The level of the noise is
    		% controlled by parameter fb3Gain.
    		% 4: speech-modulated noise. The level of the
    		% modulated noise is controlled by parameter
    		% fb4Gain; the smoothness of the intensity
    		% envelope is controlled by parameter
    		% rmsFF_fb.
    		%

            if (pert_idx == 0) % No shift
                params.fb = 1;
                params.pertAmp = zeros(size(params.pertAmp));
            elseif (pert_idx > 0 && pert_idx <= 10)
                params.fb = 1;
                params.pertAmp = (pert_idx / 10.0) * pert_amp;
            elseif (pert_idx > 10 || pert_idx < 0)
                params.fb = 2;
                params.pertAmp = pert_amp;
            end
        end

        %
        % Audapter System Recording / Tracking and Psych Toolbox Word Display Logic:
        %

        AudapterIO('init', params);
        if USE_ADAPTIVE_AGENT
            fprintf('  [audapter] bRatioShift=%d  bShift=%d  bMelShift=%d   (expect 0/1/0)\n', ...
                    params.bRatioShift, params.bShift, params.bMelShift);
        end
        %% Start Audapt and Display Word on Screen
        Audapter('start');
        
        v1 = Screen(w, 'Flip');
        WaitSecs(WORD_DISPLAY_TIME);

        %% Word disappears ...
        Screen('FillRect', w, black);
        v2 = Screen(w, 'Flip');
        %v2 - v1 % Is typically 2.005 s
        WaitSecs(AUDIO_INTAKE_TIME);
        Audapter('stop');
        B_VIS = 1;
        B_VIS_FORMATS = 1;
        VIS_NAME = 'Persistent formant shift';

        %% --Write WAV and DATA Files
        %wavwrite(2*resample(sig(:, 1),fs*4,fs), fs*4,'test1.wav');
        %wavwrite(2*resample(sig(:, 2),fs*4,fs), fs*4,'test2.wav');

        % Grab the output data from the Audapter formant tracking system:
        data = AudapterIO('getData');

        %
        % @IMPORTANT This is after a single trial / stimulus / word has finished, and here we grab the output data of the trial from Audapter:
        %   Note on Audapter Data format:
        %     data.fmts contains the tracked F1 and F2 formants in a matrix with the format number by column and each sample by row, for example, data.fmts(i, 1) gives you the ith F1 sample and data.fmts(i, 2) gives you the ith F2 sample from the trial / stimulus / word
        %     data.sfmts contains a shifted version of the tracked F1 and F2 formants. These are the formants for the output signal from Audapter. Same formatting as data.fmts.
        %
        % For more information on the output data here, see the official Audapter manual
        %   The latest Audapter manual can be found here: https://sites.bu.edu/guentherlab/files/2022/09/AudapterManual_2.1.5.pdf (Although it's probably a good idea to check the version of Audapter used in this experiment because it may not be the latest Audapter version)
        % See pages 28 and 29
        %

        if run_screening && (stim_id == 0 || stim_id == num_words_to_use)
            continue; % Don't record the results of the first and last words
        end

        %
        % Data Formatting / Output Logic:
        %
        % Everything below here is logic for saving data to various matrices and files, it does not have any effect on the logic
        %   of the experiment or the program's interaction with Audapter, it simply takes the outputted Audapter data and saves them
        %   to accessible file formats for analysis after the experiment is finished (wav, fbw, txt, etc.):
        %

        % Save the input audio data to a wav file:
        audiowrite(out_names{stim_id, 1}, data.signalIn,data.params.sr);
        % Save the output audio data (audio that has been formant shifted by Audapter) to a wav file:
        audiowrite(out_names{stim_id, 2}, data.signalOut,data.params.sr);

        save(out_namesData{stim_id, 1}, 'data');

        % Creating an array to represent the time at each formant sample:
        time_data = zeros(size(data.intervals, 1), 1);
        for iter = 1 : size(data.intervals, 1)
            time_data(iter, 1) = 1000.0 * data.intervals(iter, 1)/SAMPLING_RATE * DOWNSAMPLING_FACTOR;
        end

        % Output all formant and smoothed formant data to a text file (along with their corresponding timestamps):
        formants = horzcat(time_data, data.fmts, data.sfmts);
        dlmwrite(out_namesFormData{stim_id, 1}, formants, 'delimiter', '\t', 'newline', 'pc');

        for fbw_type = 1 : 2 % fbw_type 1 means the normal fbw file, type 2 means the shifted fbw file
            if fbw_type == 1 filename = out_namesFBW{stim_id, 1};
            else filename = out_namesFBW{stim_id, 2};
            end
            fbw_id = fopen(filename,'wt+');
            for iter = 1 : size(data.intervals, 1)
                time = 1000.0 * data.intervals(iter, 1) * DOWNSAMPLING_FACTOR / SAMPLING_RATE;
                if fbw_type == 1
                    fmt1 = data.fmts(iter, 1) / 1000.0;
                    fmt2 = data.fmts(iter, 2) / 1000.0;
                else
                    fmt1 = data.sfmts(iter, 1) / 1000.0;
                    fmt2 = data.sfmts(iter, 2) / 1000.0;
                end
                fmt3 = data.fmts(iter, 3) / 1000.0;
                db_value = -100.0;
                pitch_duration = 0.0;
                if (fmt1 > 1.0e-3) %@REVISE?
                    db_value = -35.0; %@REVISE?
                    if (iter > 1) pitch_duration = time - prev_time; end
                end
                prev_time = time;

                fprintf(fbw_id, '%.3f\t%.3f\t%.3f\t%.3f\t%.1f\t%.3f\t0\n', time, fmt1, ...
                    fmt2, fmt3, db_value, pitch_duration);
            end
            fclose(fbw_id);
        end

        %%
        %% Finds the longest interval of formant tracking which is represented by vowel_end_time - vowel_start_time:
        %%   This allows us to define the start and end times of different vowels, so we can automatically label
        %%   where the vowels are in the LBL files:
        %%
        NUM_EMPTY_TIMES_TO_STOP = 8;
        MIN_VOWEL_INTERVAL = 40.0;
        MIN_FORMANT_FREQUENCY = 1.0e-3;
        started_reading_interval = false;
        num_empty_times = 0;
        num_vowel_times = 0;

        %@NOTE: For the vectors below, the first component is the start
        %time/index, the second component is the end time/index
        vowel_times = zeros(1, 2);
        vowel_time_indices = zeros(1, 2);
        max_vowel_time = zeros(1, 2);
        max_vowel_time_indices = zeros(1, 2);

        should_record_interval = false;
        num_intervals = size(data.intervals, 1);
        for iter = 2 : num_intervals
            if data.fmts(iter, 1) > MIN_FORMANT_FREQUENCY
                num_empty_times = 0;
                if ~started_reading_interval
                    started_reading_interval = true;
                    start_time = 1000.0 * data.intervals(iter, 1) * DOWNSAMPLING_FACTOR / SAMPLING_RATE;
                    start_index = iter;
                end
            else
                if started_reading_interval
                    num_empty_times = num_empty_times + 1;
                    if num_empty_times >= NUM_EMPTY_TIMES_TO_STOP should_record_interval = true; end
                end
            end

            if iter == num_intervals && started_reading_interval % The vowel ends at the very end of the recording
                should_record_interval = true;
            end

            if should_record_interval
                end_time = 1000.0 * data.intervals(iter - num_empty_times, 1) * DOWNSAMPLING_FACTOR / SAMPLING_RATE;
                end_index = iter;
                interval = end_time - start_time;
                if interval >= MIN_VOWEL_INTERVAL
                    num_vowel_times = num_vowel_times + 1;
                    vowel_times(num_vowel_times, 1) = start_time;
                    vowel_times(num_vowel_times, 2) = end_time;
                    vowel_time_indices(num_vowel_times, 1) = start_index;
                    vowel_time_indices(num_vowel_times, 2) = end_index;
                end
                if interval > max_vowel_time(1, 2) - max_vowel_time(1, 1)
                    max_vowel_time(1, 1) = start_time;
                    max_vowel_time(1, 2) = end_time;
                    max_vowel_time_indices(1, 1) = start_index;
                    max_vowel_time_indices(1, 2) = end_index;
                end
                started_reading_interval = false;
                num_empty_times = 0;
                should_record_interval = false;
            end
        end

        if SINGLE_VOWEL_MODE
            if max_vowel_time(1, 2) > 0
                num_vowel_times = 1;
                vowel_times(1, 1) = max_vowel_time(1, 1);
                vowel_times(1, 2) = max_vowel_time(1, 2);
                vowel_time_indices(1, 1) = max_vowel_time_indices(1, 1);
                vowel_time_indices(1, 2) = max_vowel_time_indices(1, 2);
            else
                num_vowel_times = 0;
            end
        end

        %@REVISE?:
        %{
        [interval_start, interval_end] = getFmtPlotBounds(data.fmts(:, 1), data.fmts(:, 2));
        time_start = 1000.0 * data.intervals(interval_start, 1) * DOWNSAMPLING_FACTOR / SAMPLING_RATE;
        time_end = 1000.0 * data.intervals(interval_end, 1) * DOWNSAMPLING_FACTOR / SAMPLING_RATE;
        
        vowel_times(1, 1) = time_start;
        vowel_times(1, 2) = time_end;
        vowel_time_indices(1, 1) = interval_start;
        vowel_time_indices(1, 2) = interval_end;
        num_vowel_times = 1;
        %}

        %% Stat calculations and file outputs:
        if num_vowel_times > 0
            mid_index = int32(vowel_time_indices(1, 1) + (vowel_time_indices(1, 2) - vowel_time_indices(1, 1))/2);
            avgs = zeros(2, 1);
            s_avgs = zeros(2, 1);
            WINDOW_SIZE = 20;
            half = int32(WINDOW_SIZE / 2);
            % Clamp the averaging window to valid sample indices (guards against
            % a vowel detected near the start/end of the recording) and divide by
            % the number of samples actually summed (fixes the prior 21/20 bias).
            lo = max(int32(1), mid_index - half);
            hi = min(int32(size(data.fmts, 1)), mid_index + half);
            n_win = double(hi - lo + 1);
            for iter = 1 : 2
                for iter2 = lo : hi
                    avgs(iter, 1) = avgs(iter, 1) + data.fmts(iter2, iter);
                    s_avgs(iter, 1) = s_avgs(iter, 1) + data.sfmts(iter2, iter);
                end
                avgs(iter, 1) = avgs(iter, 1) / n_win;
                s_avgs(iter, 1) = s_avgs(iter, 1) / n_win;
            end
            avg_mid_fmt_data(stim_id, 1) = avgs(1, 1);
            avg_mid_fmt_data(stim_id, 2) = avgs(2, 1);
            avg_mid_sfmt_data(stim_id, 1) = s_avgs(1, 1);
            avg_mid_sfmt_data(stim_id, 2) = s_avgs(2, 1);

            obs_f1 = avgs(1, 1);  obs_f2 = avgs(2, 1);
            % Plausibility check on the measured formants (the bad-take defense):
            valid_take = (obs_f1 >= VOWEL_F1_MIN && obs_f1 <= VOWEL_F1_MAX) && ...
                (obs_f2 >= VOWEL_F2_MIN && obs_f2 <= VOWEL_F2_MAX) && ...
                (obs_f2 > obs_f1);
            if ~valid_take
                fprintf('  [agent] IMPLAUSIBLE take obs=(%.0f,%.0f) -> rejected (treated as no-vowel)\n', ...
                    obs_f1, obs_f2);
            end

            if USE_ADAPTIVE_AGENT && valid_take
                if stim_id <= ADAPTIVE_NUM_BASELINE
                    % --- Baseline phase: keep feedback unshifted, accumulate a stable anchor ---
                    baseline_sum   = baseline_sum + [obs_f1, obs_f2];
                    baseline_count = baseline_count + 1;
                    next_pert_amp  = zeros(1, 257);
                    next_pert_phi  = zeros(1, 257);
                    fprintf('  [agent] BASELINE %d/%d  obs=(%.0f,%.0f)\n', ...
                        stim_id, ADAPTIVE_NUM_BASELINE, obs_f1, obs_f2);
                end

                % --- log this trial (obs / shift / applied action) ---
                log_obs(stim_id, :)  = [obs_f1, obs_f2];
                log_sobs(stim_id, :) = [s_avgs(1,1), s_avgs(2,1)];
                log_dist(stim_id)    = sqrt((obs_f1-TARGET_U_F1)^2 + (obs_f2-TARGET_U_F2)^2);
                if stim_id <= ADAPTIVE_NUM_BASELINE, log_phase(stim_id) = 1;
                else,                                 log_phase(stim_id) = 2; end
                if agent_primed && exist('next_action','var')
                    log_action(stim_id, :) = next_action(:).';
                end

                if ~agent_primed && stim_id >= ADAPTIVE_NUM_BASELINE
                    % --- End of baseline: hand the AVERAGED anchor to the agent for its first action ---
                    if baseline_count > 0
                        anchor = baseline_sum / baseline_count;
                    else
                        anchor = [obs_f1, obs_f2];   % fallback if no valid baseline was captured
                    end
                    result = py.agent_bridge.act(anchor(1), anchor(2));
                    agent_primed = true;
                    next_action = cellfun(@double, cell(result));   % [dF1, dF2] in Hz
                    amp = norm(next_action);
                    phi = atan2(next_action(2), next_action(1));
                    if phi < 0, phi = phi + 2*pi; end
                    next_pert_amp = (PERT_HZ_TO_AMP * amp) * ones(1, 257);
                    next_pert_phi = phi * ones(1, 257);
                    fprintf('  [agent] baseline anchor=(%.0f,%.0f) over %d trials -> first action=(%.1f,%.1f)\n', ...
                        anchor(1), anchor(2), baseline_count, next_action(1), next_action(2));
                    est = cellfun(@double, cell(py.agent_bridge.get_estimate()));
                    fprintf('          est k=(%.3f,%.3f)  cross=(%.3f,%.3f)  (prior; pre-identification)\n', ...
                        est(1), est(2), est(3), est(4));
                elseif agent_primed
                    % --- Control phase: normal closed-loop step ---
                    result = py.agent_bridge.act(obs_f1, obs_f2);
                    next_action = cellfun(@double, cell(result));   % [dF1, dF2] in Hz
                    amp = norm(next_action);
                    phi = atan2(next_action(2), next_action(1));
                    if phi < 0, phi = phi + 2*pi; end
                    next_pert_amp = (PERT_HZ_TO_AMP * amp) * ones(1, 257);
                    next_pert_phi = phi * ones(1, 257);
                    fprintf('  [agent] obs=(%.0f,%.0f)  target=(%.0f,%.0f)  action=(%.1f,%.1f)  applied_shift=(%.1f,%.1f)\n', ...
                        obs_f1, obs_f2, TARGET_U_F1, TARGET_U_F2, next_action(1), next_action(2), ...
                        s_avgs(1,1)-avgs(1,1), s_avgs(2,1)-avgs(2,1));
                    est = cellfun(@double, cell(py.agent_bridge.get_estimate()));
                    fprintf('          est k=(%.3f,%.3f)  cross=(%.3f,%.3f)\n', ...
                        est(1), est(2), est(3), est(4));
                end

                % --- estimate as of THIS trial's update (post-act); captures trial-30 too ---
                if agent_primed && exist('est','var')
                    log_estk(stim_id, :)  = [est(1), est(2)];
                    log_cross(stim_id, :) = [est(3), est(4)];
                end
            end
        end
            if num_vowel_times > 0 % A vowel interval was found
                lbl_file_id = fopen(out_namesLBL{stim_id, 1}, 'wt+');
                lblr_file_id = fopen(out_namesLBL{stim_id, 2}, 'wt+');
                for iter = 1 : num_vowel_times
                    label_name = sprintf('%02d', iter);
                    fprintf(lbl_file_id, '%f\t%f\t%s\n', vowel_times(iter, 1), vowel_times(iter, 2), label_name);
                    fprintf(lblr_file_id, '%f\t%f\t%s\n', vowel_times(iter, 1), vowel_times(iter, 2), label_name);
                end
                fclose(lbl_file_id);
                fclose(lblr_file_id);
            end
            % Skip if no vowel was detected OR the take was implausible. In both cases
            % we kept no measurement, so invalidate the pending pairing and keep the estimate.
            take_was_used = (num_vowel_times > 0) && exist('valid_take','var') && valid_take;
            if USE_ADAPTIVE_AGENT && ~take_was_used && agent_primed
                % No valid vowel this trial: invalidate the pending one-step pairing so the
                % next observation isn't blamed on a stale action. Keeps the learned estimate.
                py.agent_bridge.skip_trial();
                fprintf('  [agent] no vowel detected -> skip_trial (estimate kept, pairing reset)\n');
            end
        end % The end of the loop over all stimuli / words for the experiment

    	%
    	% @IMPORTANT: The code below executes AFTER all trials / stimuli / words of the experiment ARE FINISHED being executed / recorded
    	%   The code below records general information about the entire experiment including statistics in a csv format:
    	%

    	% Averages per word:
    	% This will run through every stimulus / trial and calculate averages for each unique word (taking into account the repeated iterations of the same word):
        avg_fmt_values = zeros(NUM_UNIQUE_STIM_WORDS, 2);
        avg_sfmt_values = zeros(NUM_UNIQUE_STIM_WORDS, 2);
        min_fmt_values = realmax*ones(1, 2);
        min_sfmt_values = realmax*ones(1, 2);
        max_fmt_values = -realmax*ones(1, 2);
        max_sfmt_values = -realmax*ones(1, 2);
        fmt_value_ranges = zeros(1, 2);
        sfmt_value_ranges = zeros(1, 2);
        for iter = 1 : NUM_UNIQUE_STIM_WORDS % We iterate over each unique word:
            for iter2 = 1 : NUM_WORD_ITERATIONS % Each unique word has a certain number of iterations:
    			% The index of the jth iteration of the ith unique word is: i + NUM_UNIQUE_STIM_WORDS*(j - 1)
    			% This is because the full sequence of stim words is repeated 'NUM_WORD_ITERATIONS' number of times
    			% @REVISE @BUG @CLEANUP
    			% @REVISE @BUG @CLEANUP
    			% @REVISE @BUG @CLEANUP: It looks like this may only work for the screening and not for non-screening experiments, this may want to be checked...
    			%   If it does indeed not work for non-screening experiments, we would need to have an array containing the averages for each word (rather than each stimulus / trial), and we
    			%   would loop through each stimulus / trial and accumulate each one to the running average for the matching word in the array
    			%   (so duplicate stimuli of a word will have their averages calculated correctly regardless of the flow / order of the experiment):
    			%   -pberry 1/12/2026
                fmt_data_index = iter + NUM_UNIQUE_STIM_WORDS*(iter2 - 1);
                for fmt_num = 1 : 2
                    avg_fmt_values(iter, fmt_num) = avg_fmt_values(iter, fmt_num) + avg_mid_fmt_data(fmt_data_index, fmt_num); %@CHECK!!!
                    min_fmt_values(1, fmt_num) = min(min_fmt_values(1, fmt_num), avg_mid_fmt_data(fmt_data_index, fmt_num));
                    max_fmt_values(1, fmt_num) = max(max_fmt_values(1, fmt_num), avg_mid_fmt_data(fmt_data_index, fmt_num));

                    avg_sfmt_values(iter, fmt_num) = avg_sfmt_values(iter, fmt_num) + avg_mid_sfmt_data(fmt_data_index, fmt_num);
                    min_sfmt_values(1, fmt_num) = min(min_sfmt_values(1, fmt_num), avg_mid_sfmt_data(fmt_data_index, fmt_num));
                    max_sfmt_values(1, fmt_num) = max(max_sfmt_values(1, fmt_num), avg_mid_sfmt_data(fmt_data_index, fmt_num));
                end
            end
            for fmt_num = 1 : 2
                avg_fmt_values(iter, fmt_num) = avg_fmt_values(iter, fmt_num) / NUM_WORD_ITERATIONS;
                avg_sfmt_values(iter, fmt_num) = avg_sfmt_values(iter, fmt_num) / NUM_WORD_ITERATIONS;
            end
        end
        for fmt_num = 1 : 2
            fmt_value_ranges(1, fmt_num) = max_fmt_values(1, fmt_num) - min_fmt_values(1, fmt_num);
            sfmt_value_ranges(1, fmt_num) = max_sfmt_values(1, fmt_num) - min_sfmt_values(1, fmt_num);
        end
        if run_screening
            for iter = 1 : NUM_VOWEL_CALCULATIONS
                index1 = VOWEL_CALCULATION_INDICES(iter, 1);
                %@CHECK: Sign on these:
                dfmt1 = -PERCENT_FORMANT_CHANGE(iter, 1)*fmt_value_ranges(1, 1);
                dfmt2 = -PERCENT_FORMANT_CHANGE(iter, 2)*fmt_value_ranges(1, 2);

                pert_phi_values(iter, 1) = atan2(dfmt2, dfmt1);
                if pert_phi_values(iter, 1) < 0 pert_phi_values(iter, 1) = pert_phi_values(iter, 1) + 2*pi; end
                pert_phi_values(iter, 1) = pert_phi_values(iter, 1) / pi;
                dist = sqrt(dfmt1*dfmt1 + dfmt2*dfmt2);
                euclidean_dist_values(iter, 1) = dist;
                pert_amp_values(iter, 1) = dist / avg_fmt_values(index1, 2); %@CHECK!!!
            end
        end

    	% Outputting a csv file with various statistics about the trials (We output in Unicode so we can use the special greek symbols for vowels):
        fid = fopen('stats.csv', 'wt+', 'n', 'UTF-8');
        fprintf(fid, '%s', sprintf('\xFEFF')); % Unicode byte order mark
        fprintf(fid, 'Vowel,Word,Trial,F1,F2,SF1,SF2,Average F1,Average F2,Average SF1,Average SF2,');
        if run_screening
            for iter = 1 : NUM_VOWEL_CALCULATIONS
                index1 = VOWEL_CALCULATION_INDICES(iter, 1);
                index2 = VOWEL_CALCULATION_INDICES(iter, 2);
                fprintf(fid, 'Euclidean Dist From %s to %s,', SCREENING_VOWEL_NAMES{1, ...
                    index1}, SCREENING_VOWEL_NAMES{1, index2});
                fprintf(fid, 'Pert Amp From %s to %s,', SCREENING_VOWEL_NAMES{1, index1}, SCREENING_VOWEL_NAMES{1, index2});
                fprintf(fid, 'Phi Angle From %s to %s (In ''Radians / PI''),', SCREENING_VOWEL_NAMES{1, ...
                    index1}, SCREENING_VOWEL_NAMES{1, index2});
            end
        end
        fprintf(fid, '\n');
        for iter = 1 : NUM_UNIQUE_STIM_WORDS
            fprintf(fid, '%s,%s,1,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,', SCREENING_VOWEL_NAMES{1, iter}, stim_data{iter, 1}, avg_mid_fmt_data(iter, 1), avg_mid_fmt_data(iter, 2), ...
                avg_mid_sfmt_data(iter, 1), avg_mid_sfmt_data(iter, 2), avg_fmt_values(iter, 1), avg_fmt_values(iter, 2), avg_sfmt_values(iter, 1), avg_sfmt_values(iter, 2));
            if iter == 1 && run_screening
                for iter2 = 1 : NUM_VOWEL_CALCULATIONS
                    fprintf(fid, '%.0f,%.4f,%.4f,', euclidean_dist_values(iter2, 1), pert_amp_values(iter2, 1), ...
                        pert_phi_values(iter2, 1));
                end
            end
            fprintf(fid, '\n');
            for iter2 = 2 : NUM_WORD_ITERATIONS
                fmt_data_index = iter + NUM_UNIQUE_STIM_WORDS*(iter2 - 1);
                fprintf(fid, ',,%d,%.0f,%.0f,%.0f,%.0f\n', iter2, avg_mid_fmt_data(fmt_data_index, 1), avg_mid_fmt_data(fmt_data_index, 2), ...
                    avg_mid_sfmt_data(fmt_data_index, 1), avg_mid_sfmt_data(fmt_data_index, 2));
            end
        end
        fprintf(fid, ',,MIN,%.0f,%.0f,%.0f,%.0f\n,,MAX,%.0f,%.0f,%.0f,%.0f\n,,RANGE,%.0f,%.0f,%.0f,%.0f\n', min_fmt_values(1, 1), min_fmt_values(1, 2), ...
            min_sfmt_values(1, 1), min_sfmt_values(1, 2), max_fmt_values(1, 1), max_fmt_values(1, 2), ...
            max_sfmt_values(1, 1), max_sfmt_values(1, 2), fmt_value_ranges(1, 1), fmt_value_ranges(1, 2), sfmt_value_ranges(1, 1), ...
            sfmt_value_ranges(1, 2));

        fclose(fid);

        break;
    end

    %@NECESSARY?:
    KbQueueStop(0);
    KbQueueRelease(0);
    Screen('CloseAll');

    save('AdaptationRun', 'idx_data', 'out_names', 'out_namesData', 'out_namesFormData', ...
        'stim_data', 'stim_id');

    %% -- Visualization (Graphs) -- %%
    if B_VIS
        data = AudapterIO('getData');

        frame_dur = data.params.frameLen / data.params.sr;
        t_axis = 0 : frame_dur : frame_dur * (size(data.fmts, 1) - 1);

        %-----------------------%
        figure;
        subplot('Position', [0.1, 0.5, 0.8, 0.375]);
        show_spectrogram(data.signalIn, data.params.sr, 'noFig');

        if B_VIS_FORMATS
            plot(t_axis, data.fmts(:, 1 : 2), 'Color', GRAY);
        end
        if B_VIS_OST
            plot(t_axis, data.ost_stat * OST_MULT, 'b-');
        end

        ylabel('Frequency (Hz)');

        xs = get(gca, 'XLim');
        ys = get(gca, 'YLim');
        text(xs(1) + 0.025 * (max(xs)-min(xs)), ys(2) - 0.075 * (max(ys)-min(ys)), ...
            'Input sound', 'FontSize', 12);

        %-----------------------%
        subplot('Position', [0.1, 0.125, 0.8, 0.375]);
        hold on;
        show_spectrogram(data.signalOut, data.params.sr, 'noFig');

        legend_items = {};
        if B_VIS_FORMATS
            plot(t_axis, data.fmts(:, 1 : 2), 'Color', GRAY);
            plot(t_axis, data.sfmts(:, 1 : 2), 'g');
            legend_items{end + 1} = 'Original F1';
            legend_items{end + 1} = 'Original F2';
            legend_items{end + 1} = 'Shifted F1';
            legend_items{end + 1} = 'Shifted F2';
        end
        if B_VIS_OST
            plot(t_axis, data.ost_stat * OST_MULT, 'b-');
            legend_items{end + 1} = sprintf('OST stat * %d', OST_MULT);
        end

        xlabel('Time (s)');
        ylabel('Frequency (Hz)');

        xs = get(gca, 'XLim');
        ys = get(gca, 'YLim');
        text(xs(1) + 0.025 * (max(xs)-min(xs)), ys(2) - 0.075 * (max(ys)-min(ys)), ...
            sprintf('Output sound: %s', VIS_NAME), 'FontSize', 12);

        if ~isempty(legend_items)
            legend(legend_items, 'FontSize', LEGEND_FONT_SIZE, ...
                'Location', 'Southwest');
        end
    end

  if USE_ADAPTIVE_AGENT
        % --- Assemble the diagnostic struct from the per-trial logs ---
        % realized shift = shifted formants minus measured formants (signed, Hz):
        %   +ve means Audapter raised that formant in the feedback the subject heard.
        log.trial = (1:ADAPTIVE_NUM_TRIALS).';
        log.obs   = log_obs;
        log.act   = log_action;
        log.shift = log_sobs - log_obs;
        log.k     = log_estk;
        log.cross = log_cross;
        log.base  = (log_phase == 1);
 
        % SAVE FIRST -- a plotting hiccup must never cost you the subject's data.
        save(sprintf('adaptation_log_%s.mat', subject_id), ...
            'log','log_obs','log_sobs','log_action','log_estk','log_cross','log_dist', ...
            'log_phase','TARGET_U_F1','TARGET_U_F2','ADAPTIVE_WORD', ...
            'ADAPTIVE_NUM_BASELINE','ADAPTIVE_NUM_TRIALS');
        fprintf('Wrote adaptation_log_%s.mat\n', subject_id);
 
        % Plot is best-effort: wrapped so a failure warns instead of aborting.
        if ~any(~isnan(log_dist))
            warning('No valid trials logged -- nothing to plot (check tracking / rmsThresh).');
        else
            try
                plot_adaptation_diagnostics(log, [TARGET_U_F1, TARGET_U_F2], ADAPTIVE_NUM_BASELINE);
                saveas(gcf, sprintf('adaptation_summary_%s.png', subject_id));
                fprintf('Wrote adaptation_summary_%s.png\n', subject_id);
            catch ME
                warning('Plotting failed (%s). Raw log was saved; re-plot offline from the .mat.', ME.message);
            end
        end
    end
 
    cd('../..');
end