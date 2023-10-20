#!/usr/bin/env python

import sys

next_expected_next_block = 0
expected_num_audio_in_block = None  # set from first
num_audio_in_last_block = 0

num_ctrls_recorded = None  # set from first
num_audio_ins_records = None  # set from first

latest_ctrl_values = None  # these are records before each audio

for line_num, line in enumerate(sys.stdin):
  line = line.strip()
  # ignore headers
  if line == 'Daisy is online': continue
  if line == '===============': continue
  if 'wr' in line: continue
  # new block?
  if line.startswith('b'):
    # check blocks are going up
    block_id = int(line.split(" ")[1])
    if block_id != next_expected_next_block:
      raise Exception(f"next_expected_next_block={next_expected_next_block}"
                      f" but got block_id={block_id}")
    next_expected_next_block += 1

    # check audio block sizes are consistent
    # all blocks must match size of block0
    if block_id == 0:
      pass
    elif block_id == 1:
      # take expected count from last block, i.e. block 0
      assert num_audio_in_last_block > 0, num_audio_in_last_block
      expected_num_audio_in_block = num_audio_in_last_block
    else:
      assert num_audio_in_last_block == expected_num_audio_in_block, block_id
    num_audio_in_last_block = 0
    latest_ctrl_values = None

  elif line.startswith('c'):
    cols = line.split(" ")
    first_col = cols.pop(0)
    assert first_col == 'c'
    if num_ctrls_recorded is None:
      num_ctrls_recorded = len(cols)
    else:
      assert len(cols) == num_ctrls_recorded
    _ = map(float, cols)  # just a check for parsable floats
    latest_ctrl_values = cols

  elif line.startswith('a'):
    assert latest_ctrl_values is not None
    num_audio_in_last_block += 1
    cols = line.split(" ")
    first_col = cols.pop(0)
    assert first_col == 'a'
    if num_audio_ins_records is None:
      num_audio_ins_records = len(cols)
    else:
      assert len(cols) == num_audio_ins_records
    _ = map(float, cols)  # just a check for parsable floats
    #print(" ".join(latest_ctrl_values + cols))
    print(" ".join(cols))

  else:
    raise Exception(f"unexpected line [{line}]")
