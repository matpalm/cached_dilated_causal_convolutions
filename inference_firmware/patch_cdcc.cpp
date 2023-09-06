#include <string>

#include "arm_math.h"
#include "daisy_patch.h"
#include "daisysp.h"

#include "model_defn.h"

using namespace daisy;
using namespace daisysp;
using namespace std;

DaisyPatch hw;
CpuLoadMeter cpu_load_meter;

bool foo = true;
long inference_calls = 0;

void WriteArray(string msg, float* a, size_t n) {
  FixedCapStr<100> str;
  str.Append(">>>>>>> [");
  char* cstr = &msg[0];
  str.Append(cstr);
  str.Append("] n=");
  str.AppendInt(n);
  hw.seed.PrintLine(str);
  for (size_t i=0; i<n; i++) {
    str.Clear();
    str.AppendInt(i);
    str.Append(" ");
    str.AppendFloat(a[i], 5);
    hw.seed.PrintLine(str);
  }
}

void Write2DArray(string msg, float* a, size_t n, size_t m) {
  FixedCapStr<100> str;
  str.Append(">>>>>>> [");
  char* cstr = &msg[0];
  str.Append(cstr);
  str.Append("] n=");
  str.AppendInt(n);
  str.Append(" m=");
  str.AppendInt(m);
  hw.seed.PrintLine(str);
  for (size_t i=0; i<n; i++) {
    for (size_t j=0; j<m; j++) {
      str.Clear();
      str.Append(" ");
      str.AppendFloat(a[(i*m)+j], 5);
      hw.seed.Print(str);
    }
    hw.seed.PrintLine("");
  }
}

FixedCapStr<100> assert_failed_msg;
bool assert_failed = false;

void AssertSame(string msg, size_t a, size_t b) {
  if (a == b) {
    // LGTM
    return;
  }
  if (assert_failed) {
    // already another failure! keep existing message
    return;
  }
  assert_failed_msg.Clear();
  assert_failed_msg.Append("AssertSame FAILDOG [");
  char* cstr = &msg[0];
  assert_failed_msg.Append(cstr);
  assert_failed_msg.Append("] a=");
  assert_failed_msg.AppendInt(a);
  assert_failed_msg.Append(" b=");
  assert_failed_msg.AppendInt(b);
  assert_failed = true;
}


void RunInference(float* next_inputs) {

  left_shift_input_buffer.Add(next_inputs);
  block0.Run();
  layer0_cache.Run();
  block1.Run();
  layer1_cache.Run();
  block2.Run();
  layer2_cache.Run();
  block3.Run();
  classifier.Run();

  inference_calls++;

}

void AudioCallback(AudioHandle::InputBuffer in,
                   AudioHandle::OutputBuffer out,
                   size_t size) {

  cpu_load_meter.OnBlockStart();

  // setup input for network, for now input is just the audio
  float next_inputs[1];
  //ctrl0_val = hw.controls[0].Value();
  //next_inputs[0] = ctrl0_val;
  // take a ptr to the classifier output for copying to output buffers
  float* classifier_out = classifier.GetOutputBuffer();
  // run the blocks
  for (size_t b = 0; b < size; b++) {
    next_inputs[0] = in[0][b];
    RunInference(next_inputs);
    out[0][b] = in[0][b];
    out[1][b] = classifier_out[0];
    out[2][b] = classifier_out[1];
  }

  cpu_load_meter.OnBlockEnd();
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

  FixedCapStr<12> str("cpu ");
  const float cpu = cpu_load_meter.GetAvgCpuLoad();
  str.AppendFloat(cpu, 5);
  strs.push_back(string(str));
  //hw.seed.PrintLine(str);

  // str.Clear();
  // str.Append("ctrl0 ");
  // str.AppendFloat(ctrl0_val, 3);
  // strs.push_back(string(str));

  str.Clear();
  str.AppendInt(inference_calls);
  strs.push_back(string(str));

  if (assert_failed) {
    hw.seed.PrintLine(assert_failed_msg);
  }

  DisplayLines(strs);
  hw.display.Update();

  //RunInference();
  //Write2DArray("classifier.in", classifier.GetInputBuffer(), 1, 8);
  //Write2DArray("classifier.out", classifier.GetOutputBuffer(), 1, 2);

  //hw.seed.DelayMs(10);  // ms
}

int main(void) {

  // assertions regarding shapes
  AssertSame("inp->b0",
    left_shift_input_buffer.GetOutputBufferSize(),
    block0.GetInputBufferSize()
  );
  AssertSame("b0->l0",
    block0.GetOutputBufferSize(),
    layer0_cache.GetInputBufferSize()
  );
  AssertSame("l0->b1",
    layer0_cache.GetOutputBufferSize(),
    block1.GetInputBufferSize()
  );
  AssertSame("b1->l1",
    block1.GetOutputBufferSize(),
    layer1_cache.GetInputBufferSize()
  );
  AssertSame("l1->b2",
    layer1_cache.GetOutputBufferSize(),
    block2.GetInputBufferSize()
  );
  AssertSame("b2->l2",
    block2.GetOutputBufferSize(),
    layer2_cache.GetInputBufferSize()
  );
  AssertSame("l2->b3",
    layer2_cache.GetOutputBufferSize(),
    block3.GetInputBufferSize()
  );
  AssertSame("b3->c",
    block3.GetOutputBufferSize(),
    classifier.GetInputBufferSize()
  );

  // connect steps
  // TODO: introduce virtual HasGetInputBuffer & HasSetOutputBuffer here ?
  left_shift_input_buffer.SetOutputBuffer(block0.GetInputBuffer());
  block0.SetOutputBuffer(layer0_cache.GetInputBuffer());
  layer0_cache.SetOutputBuffer(block1.GetInputBuffer());
  block1.SetOutputBuffer(layer1_cache.GetInputBuffer());
  layer1_cache.SetOutputBuffer(block2.GetInputBuffer());
  block2.SetOutputBuffer(layer2_cache.GetInputBuffer());
  layer2_cache.SetOutputBuffer(block3.GetInputBuffer());
  block3.SetOutputBuffer(classifier.GetInputBuffer());

  hw.Init();
  hw.SetAudioBlockSize(64); // number of samples handled per callback
  hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_32KHZ);
  //hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);
  hw.StartAdc();

  hw.seed.StartLog();
  cpu_load_meter.Init(hw.AudioSampleRate(), hw.AudioBlockSize());

  hw.StartAudio(AudioCallback);

  while(1) {
    hw.ProcessAllControls();
    UpdateDisplay();
  }

}

