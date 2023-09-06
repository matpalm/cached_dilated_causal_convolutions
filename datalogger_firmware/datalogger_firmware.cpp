#include <string>
#include "daisy_patch.h"
#include "daisysp.h"

/*

firmware for logging training data for patch_cdcc

starts in a waiting state, to ensure serial connection setup
when encoder pressed records ctrl values and audio in to buffers
when buffer full flushes to serial out

*/

using namespace daisy;
using namespace daisysp;
using namespace std;

DaisyPatch hw;
CpuLoadMeter cpu_load_meter;

enum State {
  WAITING,     // initial powered up, waiting for encoder click
  RECORDING,   // recording sample
  FLUSHING,    // writing to serial
  DONE         // done and waiting
};
State state = WAITING;

// how many ctrl ins to record? ( logged once per audio callback )
const size_t NUM_CTRLS_RECORDED = 1;  // e.g. x0 -> y0,y1

// how many input channels to record?
const size_t NUM_CHANNELS_RECORDED = 3;  // e.g. x0 -> y0,y1

//const size_t BUFFER_SIZE = 1000; // will need tuning depending on above
const size_t BUFFER_SIZE = 20000; // will need tuning depending on above
const size_t BLOCK_SIZE = 64;
float DSY_SDRAM_BSS ctrl_vals[BUFFER_SIZE][NUM_CTRLS_RECORDED];
float DSY_SDRAM_BSS input_audio[BUFFER_SIZE][NUM_CHANNELS_RECORDED][BLOCK_SIZE];
size_t buffer_idx = 0;

// bool to denote if final dump has been flushed to serial
bool serial_flushed = false;

void AudioCallback(AudioHandle::InputBuffer in, AudioHandle::OutputBuffer out,
                    size_t size) {

  hw.ProcessAllControls();

  if (hw.encoder.RisingEdge() && state==WAITING) {
    state = RECORDING;
  }

  // cpu_load_meter.OnBlockStart();

  for (size_t b=0; b < BLOCK_SIZE; b++) {
    for (size_t c=0; c < NUM_CHANNELS_RECORDED; c++) {
      out[c][b] = in[c][b];
    }
  }

  if (state == RECORDING) {
    for (size_t c=0; c < NUM_CTRLS_RECORDED; c++) {
      ctrl_vals[buffer_idx][c] = hw.controls[c].Value();
    }
    for (size_t b=0; b < BLOCK_SIZE; b++) {
      for (size_t c=0; c < NUM_CHANNELS_RECORDED; c++) {
        input_audio[buffer_idx][c][b] = in[c][b];
      }
    }
    buffer_idx++;
    if (buffer_idx == BUFFER_SIZE) {
      state = FLUSHING;
    }
  }

  // cpu_load_meter.OnBlockEnd();
}

void DisplayLines(const vector<string> &strs) {
  int line_num = 0;
  for (string str : strs) {
    char* cstr = &str[0];
    hw.display.SetCursor(0, line_num*10);
    hw.display.WriteString(cstr, Font_7x10, true);
    line_num++;
  }
}

void UpdateDisplay() {
  hw.display.Fill(false);
  vector<string> strs;

  FixedCapStr<30> str("");

  str.Clear();
  str.Append("state ");
  switch (state) {
    case WAITING:
      str.Append("WAITING");
      break;
    case RECORDING:
      str.Append("RECORDING");
      break;
    case FLUSHING:
      str.Append("FLUSHING");
      break;
    case DONE:
      str.Append("DONE");
      break;
  }
  strs.push_back(string(str));

  str.Clear();
  str.Append("buff ");
  str.AppendInt(buffer_idx);
  strs.push_back(string(str));

  float proportion_buffer_idx = float(buffer_idx) / BUFFER_SIZE;
  str.Clear();
  str.Append("buff p ");
  str.AppendFloat(proportion_buffer_idx);
  strs.push_back(string(str));

  DisplayLines(strs);
  hw.display.Update();

  if (state == WAITING || state == RECORDING) {
    hw.seed.PrintLine("wr");

  } else if (state == FLUSHING) {
    FixedCapStr<100> str("");
    for (size_t i=0; i<BUFFER_SIZE; i++) {

      str.Clear();
      str.Append("b ");
      str.AppendInt(i);
      hw.seed.PrintLine(str);

      str.Clear();
      str.Append("c");
      for (size_t c=0; c<NUM_CTRLS_RECORDED; c++) {
        str.Append(" ");
        str.AppendFloat(ctrl_vals[i][c], 9);
      }
      hw.seed.PrintLine(str);

      for (size_t b=0; b<BLOCK_SIZE; b++) {
        str.Clear();
        str.Append("a");
        for (size_t c=0; c<NUM_CHANNELS_RECORDED; c++) {
          str.Append(" ");
          str.AppendFloat(input_audio[i][c][b], 9);
        }
        hw.seed.PrintLine(str);
      }

    }
    state = DONE;

  } else if (state == DONE) {
    if (!serial_flushed) {
      hw.seed.PrintLine("");
      serial_flushed = true;
    }
  }

}


int main(void) {
  hw.Init();
  hw.SetAudioBlockSize(BLOCK_SIZE); // number of samples handled per callback
  hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_32KHZ);
  hw.StartAdc();

  // Enable Logging, and set up the USB connection.
  hw.seed.StartLog();

  cpu_load_meter.Init(hw.AudioSampleRate(), hw.AudioBlockSize());
  hw.StartAudio(AudioCallback);

  while(1) {
    hw.ProcessAllControls();
    UpdateDisplay();
  }
}

