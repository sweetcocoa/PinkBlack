import pretty_midi
import IPython.display
import numpy as np


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.

    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.

    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.

    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


def show_midi(pm, roll=False):
    """
    Piano roll 이나 PrettyMidi 인스턴스를 jupyter notebook상의 셀에서 재생 가능한 형태로 보여줍니다.
    (Show Piano Roll or PrettyMidi instances that can be played in a cell in the jupyter notebook.)

    :param pm:
    :param roll:
        parameter인 pm이 Piano roll인지를 나타냅니다. 꼭 필요한 것은 아닙니다.
        (Indicates whether the parameter pm is a Piano roll. It is not necessary.)
        Default : False
    :return:
        None.
    """
    if not roll and isinstance(pm, pretty_midi.PrettyMIDI):
        IPython.display.display(IPython.display.Audio(pm.synthesize(fs=16000), rate=16000))
    elif roll:
        new_pm = piano_roll_to_pretty_midi(pm)
        IPython.display.display(IPython.display.Audio(new_pm.synthesize(fs=16000), rate=16000))
    else:
        print("show_midi Warning : unexpected arguments")
        new_pm = piano_roll_to_pretty_midi(pm)
        IPython.display.display(IPython.display.Audio(new_pm.synthesize(fs=16000), rate=16000))
