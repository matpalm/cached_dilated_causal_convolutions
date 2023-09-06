import sys

NUM_AUDIO_CHANNELS_RECORDED = 3

next_expected_next_block = 0
expected_num_audio_in_block = None  # set from first
num_audio_in_last_block = 0
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

  elif line.startswith('c'):
    # ignore for now, see prototype notebook for parsing and join with
    # audio values
    pass

  elif line.startswith('a'):
    num_audio_in_last_block += 1
    cols = line.split(" ")
    assert len(cols) == NUM_AUDIO_CHANNELS_RECORDED + 1, f"{line_num} [{line}]"
    assert cols[0] == 'a'
    _ = map(float, cols[1:])  # just check parsable
    print(" ".join(cols[1:]))

  else:
    raise Exception(f"unexpected line [{line}]")
