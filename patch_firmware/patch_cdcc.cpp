#include <string>

#include "arm_math.h"
#include "daisy_patch.h"
#include "daisysp.h"

#include "left_shift_buffer.h"
#include "block.h"
#include "rolling_cache.h"
#include "classifier.h"

using namespace daisy;
using namespace daisysp;
using namespace std;

DaisyPatch hw;
CpuLoadMeter cpu_load_meter;

LeftShiftBuffer left_shift_input_buffer(
  4,   // kernel size
  2);  // feature depth

// statically defined blocks, layer caches and classifier

float b0_c1_kernel[4*2*4] = {-0.14211542904376984, 0.1799626648426056, 0.04956188425421715, -0.26349174976348877, -0.018953043967485428, -0.2918456792831421, -0.2146742045879364, -0.10696317255496979, 0.34640443325042725, 0.126132071018219, -0.18977971374988556, 0.08371320366859436, -0.29615211486816406, 0.3603057563304901, -0.20686165988445282, -0.27919939160346985, -0.04363759979605675, -0.3705694079399109, 0.059599459171295166, -0.40770408511161804, 0.3294539451599121, 0.12403134256601334, 0.3849093019962311, 0.2547760009765625, -0.17852529883384705, 0.022834667935967445, 0.08019418269395828, 0.022055182605981827, 0.6306363940238953, 0.1106245219707489, 0.3047659695148468, 0.2643809914588928};
float b0_c1_bias[4] = {0.14674681425094604, 0.02062080055475235, 0.11506416648626328, -0.03621285781264305};
float b0_c2_kernel[1*4*4] = {-0.2333688735961914, 0.9208389520645142, -0.25512057542800903, 0.40949809551239014, -0.8444052338600159, 0.7895674705505371, -0.10958516597747803, 0.792701244354248, -0.8322503566741943, 0.495100200176239, -0.07997804880142212, 0.630760133266449, -0.6013953685760498, -0.15152575075626373, -0.1369897723197937, 0.6354628801345825};
float b0_c2_bias[4] = {0.0, -0.0015300216618925333, 0.0, 0.0008893667836673558};
Block block0(4, // kernel_size
             2, 4, // in_d, out_d
             b0_c1_kernel, b0_c1_bias, b0_c2_kernel, b0_c2_bias);

float b1_c1_kernel[4*4*8] = {-0.10172015428543091, 0.23768797516822815, -0.2070120871067047, -0.28078964352607727, -0.14142628014087677, -0.3384734094142914, -0.29178035259246826, -0.337767630815506, 0.09665774554014206, 0.013769167475402355, -0.37203335762023926, 0.08231052756309509, 0.005794360768049955, -0.1188669428229332, 0.1258951723575592, -0.31249338388442993, 0.15334585309028625, -0.023237168788909912, 0.32675787806510925, 0.09761866927146912, -0.26568782329559326, 0.0938473641872406, 0.1806427538394928, -0.32623574137687683, -0.09959591925144196, -0.29863259196281433, -0.251860648393631, 0.21421585977077484, 0.3200463056564331, -0.06218462809920311, -0.21837960183620453, 0.25135400891304016, 0.23086830973625183, -0.209733247756958, 0.008163005113601685, -0.029589354991912842, 0.024679094552993774, -0.12038016319274902, 0.03228846192359924, 0.3396545946598053, -0.09354709088802338, 0.11428792029619217, 0.28558531403541565, -0.20608994364738464, -0.306326687335968, 0.09155184030532837, -0.30676600337028503, 0.12200061231851578, -0.31637269258499146, -0.035849183797836304, -0.20391370356082916, -0.17973093688488007, 0.2257682979106903, 0.061533212661743164, -0.2939833700656891, 0.0758017897605896, 0.1332099437713623, 0.2259742170572281, -0.036188337951898575, -0.01955622062087059, -0.026071950793266296, 0.039611004292964935, 0.15672853589057922, 0.150798961520195, -0.20940864086151123, 0.22072026133537292, 0.21643832325935364, 0.17465654015541077, -0.25677064061164856, -0.17765899002552032, 0.28365573287010193, 0.2562333047389984, 0.0024531870149075985, -0.3419714868068695, 0.09203389286994934, -0.24260017275810242, 0.07171982526779175, 0.3242169916629791, -0.04220457002520561, 0.008690320886671543, -0.06701162457466125, 0.22159937024116516, -0.1467113196849823, -0.2792341709136963, -0.18292583525180817, -0.11918690800666809, 0.03208184242248535, -0.2314758002758026, -0.21006682515144348, 0.11609937995672226, -0.2682643532752991, 0.29054221510887146, 0.13831494748592377, 0.2103700041770935, -0.2383742332458496, 0.19557346403598785, -0.06600871682167053, -0.039426952600479126, -0.10108709335327148, 0.10234299302101135, 0.27825966477394104, 0.17902782559394836, -0.26880955696105957, 0.005535811185836792, 0.0036795688793063164, -0.07014122605323792, -0.13436304032802582, -0.00816216878592968, -0.42492756247520447, 0.3781113028526306, 0.2694542109966278, -0.282479465007782, 0.32589176297187805, -0.26227131485939026, -0.24959810078144073, 0.3375478684902191, 0.20571556687355042, 0.017318636178970337, -0.08782416582107544, -0.11850935220718384, 0.4343699514865875, -0.36271822452545166, 0.09701105207204819, -0.33582615852355957, -0.22275616228580475, -0.04997069388628006, -0.2704423666000366, 0.06788796931505203};
float b1_c1_bias[8] = {0.015246120281517506, 0.10782167315483093, 0.017992647364735603, -0.06110020726919174, 0.10202756524085999, 0.027269750833511353, 0.004970033187419176, 0.12601183354854584};
float b1_c2_kernel[1*8*8] = {0.6124735474586487, -0.5700029134750366, -0.03982570767402649, -0.7864934206008911, -0.6571760773658752, -0.2468787431716919, 0.11370940506458282, 0.7124264240264893, -0.024747954681515694, 0.06007317453622818, -0.2583818733692169, 0.6261536478996277, 0.3771737217903137, 0.0361948199570179, 0.17842577397823334, -0.2448718398809433, 0.4930664896965027, 0.2946026921272278, 0.13437886536121368, -0.17270199954509735, 0.11203435063362122, 0.11985236406326294, 0.5795206427574158, 0.24273568391799927, -0.2023882418870926, -0.5548379421234131, -0.3779906630516052, -0.3454182744026184, -0.23717819154262543, -0.4649750292301178, 0.12113350629806519, -0.45640912652015686, -0.2827306389808655, -0.13786055147647858, -0.01882901042699814, 0.6876317262649536, 0.43698394298553467, 0.23559609055519104, 0.038737401366233826, -0.33971837162971497, 0.5242985486984253, 0.4920414388179779, 0.2070464938879013, -0.3533315062522888, -0.13045065104961395, -0.3202182352542877, 0.05186738073825836, 0.33183905482292175, -0.6947869658470154, -0.5567260980606079, 0.06341653317213058, -0.014916786924004555, -0.1051994115114212, -0.34243473410606384, -0.6130053400993347, -0.6476429104804993, -0.5100852251052856, -0.471709281206131, -0.1932310163974762, 0.5519954562187195, 0.27179649472236633, -0.26723507046699524, -0.564490556716919, 0.019167812541127205};
float b1_c2_bias[8] = {3.514930358505808e-05, -0.03384879231452942, -0.03310495615005493, 0.1353897601366043, 0.09195112437009811, -0.012616274878382683, -0.041386209428310394, 0.018690168857574463};
Block block1(4, // kernel_size
             4, 8, // in_d, out_d
             b1_c1_kernel, b1_c1_bias, b1_c2_kernel, b1_c2_bias);

float b2_c1_kernel[4*8*8] = {-0.0008094210061244667, 0.17313940823078156, -0.016917172819375992, 0.12548772990703583, 0.1744346022605896, -0.11477410048246384, 0.32226991653442383, -0.03269276022911072, -0.22644735872745514, 0.19988372921943665, 0.22801938652992249, -0.11432117223739624, -0.1532026082277298, -0.23995545506477356, 0.12547647953033447, -0.19291870296001434, -0.28730493783950806, 0.03222762048244476, -0.1782965064048767, 0.25868135690689087, -0.16497139632701874, 0.28794464468955994, -0.007029577158391476, -0.21900048851966858, 0.25049319863319397, -0.01776370406150818, -0.06826934963464737, -0.08731011301279068, 0.05984759330749512, 0.1372372955083847, 0.028854435309767723, -0.23103107511997223, -0.17394444346427917, 0.13473686575889587, 0.06935882568359375, 0.21772123873233795, -0.07551003247499466, -0.23444916307926178, -0.17800471186637878, 0.07764707505702972, 0.21747858822345734, -0.05097650736570358, 0.13667896389961243, -0.2455209493637085, -0.2798634469509125, -0.0063195666298270226, -0.12889154255390167, 0.18771368265151978, 0.200343057513237, -0.10035523027181625, 0.1928924024105072, -0.21836721897125244, -0.007751657161861658, -0.12836648523807526, 0.2812325954437256, 0.026809364557266235, -0.09011717140674591, 0.09750093519687653, 0.08318033069372177, -0.13098686933517456, 0.018159832805395126, 0.13922806084156036, 0.06476739794015884, 0.15083914995193481, 0.04109736531972885, 0.05433308333158493, 0.18099011480808258, 0.25448253750801086, -0.20743407309055328, -0.18154379725456238, 0.057394158095121384, -0.009384778328239918, 0.261997789144516, 0.2200605273246765, 0.12499471008777618, -0.20619046688079834, 0.25401678681373596, 0.10452393442392349, -0.2588230073451996, -0.08980685472488403, -0.1511634737253189, 0.3878774642944336, -0.1850421279668808, 0.37081441283226013, -0.2508694529533386, 0.11659462004899979, -0.04466187581419945, 0.033431392163038254, -0.17876489460468292, 0.07700833678245544, 0.21748988330364227, -0.09381872415542603, 0.17144916951656342, -0.13926610350608826, 0.15840256214141846, -0.22502003610134125, -0.02204027585685253, 0.20876924693584442, -0.2735423147678375, 0.04583081230521202, 0.14532166719436646, -0.08174944669008255, 0.12552593648433685, 0.014151773415505886, 0.1561155766248703, -0.15970712900161743, 0.1419336199760437, -0.20682498812675476, -0.18469391763210297, 0.048952747136354446, 0.21189436316490173, -0.05199152231216431, -0.023054441437125206, -0.16349032521247864, -0.004340107552707195, -0.030599545687437057, 0.06443840265274048, -0.007085415069013834, 0.0646957978606224, -0.1670127809047699, -0.08994933217763901, 0.3213259279727936, 0.062245048582553864, -0.17523154616355896, -0.07210366427898407, 0.2488153725862503, 0.11629848927259445, -0.08972711861133575, 0.139637753367424, -0.15226249396800995, -0.13339033722877502, 0.08578230440616608, -0.021637916564941406, 0.3556787669658661, -0.17879648506641388, -0.019027166068553925, 0.12128684669733047, -0.26299694180488586, -0.20598773658275604, 0.02304830215871334, -0.21207295358181, 0.10732543468475342, 0.15625673532485962, -0.2600776255130768, -0.2348330318927765, 0.3122914433479309, 0.12887713313102722, 0.1397732049226761, 0.3593628704547882, 0.1954505294561386, 0.0014094705693423748, -0.2180325835943222, -0.06333194673061371, -0.03785407170653343, 0.1113179475069046, -0.09252520650625229, 0.07256122678518295, 0.029151758179068565, 0.0030356012284755707, 0.08630633354187012, -0.0886334553360939, -0.08277453482151031, -0.034879181534051895, 0.1316376030445099, -0.0064803361892700195, 0.033351462334394455, 0.19595278799533844, -0.08795925974845886, -0.04201506823301315, -0.250779926776886, 0.2741678059101105, -0.16292592883110046, 0.24115493893623352, 0.28990525007247925, -0.10313421487808228, -0.030716121196746826, 0.09251091629266739, 0.1290796995162964, 0.16407063603401184, 0.007094772066920996, 0.16059964895248413, -0.25406613945961, -0.02008357085287571, 0.10375499725341797, 0.2216874659061432, 0.25713202357292175, -0.1195467934012413, -0.10070377588272095, -0.13824522495269775, -0.1451869159936905, -0.017880240455269814, 0.22629956901073456, 0.13375674188137054, 0.05896298587322235, 0.06927677243947983, -0.3689030408859253, -0.15068410336971283, -0.17783793807029724, 0.03480381518602371, -0.004590232390910387, -0.10565783083438873, 0.08904905617237091, -0.2086978554725647, -0.22254404425621033, 0.39807525277137756, -0.30647820234298706, 0.23811809718608856, 0.018543142825365067, -0.17134204506874084, -0.28407445549964905, -0.21146920323371887, -0.16460934281349182, 0.08284062147140503, 0.2052958607673645, -0.1898694932460785, -0.16372504830360413, -0.4035933315753937, 0.16872985661029816, -0.18027767539024353, 0.32397541403770447, -0.13169968128204346, 0.4104209244251251, 0.1949865221977234, -0.2639024555683136, 0.04700838401913643, 0.03305236995220184, -0.1121184304356575, 0.359904408454895, 0.029921183362603188, -0.022340601310133934, -0.04568079486489296, 0.20792008936405182, 0.011056261137127876, 0.17317849397659302, 0.09582474827766418, 0.03631366044282913, 0.2954709827899933, -0.2883596420288086, -0.16312330961227417, -0.09173795580863953, 0.1613111048936844, 0.11245226860046387, -0.2907283902168274, 0.28823795914649963, 0.2451765090227127, 0.1729653775691986, 0.08449359238147736, -0.25843334197998047, 0.2613769769668579, -0.10028026252985, 0.2232031524181366, -0.1800527572631836, 0.07736726850271225, 0.022601163014769554, 0.26370248198509216, -0.2537985146045685};
float b2_c1_bias[8] = {0.0917653813958168, 0.047953587025403976, 0.0729256346821785, 0.0911201685667038, -0.005230237729847431, 0.062005795538425446, 0.09862358868122101, -0.01906679943203926};
float b2_c2_kernel[1*8*8] = {-0.5723579525947571, 0.5809540748596191, 0.46454572677612305, 0.3709069788455963, -0.8029416799545288, -0.044163282960653305, 0.3382247984409332, 0.13849255442619324, -0.3502344787120819, -0.3392648696899414, -0.31114792823791504, 0.15653450787067413, -0.03602295368909836, -0.39755573868751526, 0.47580283880233765, 0.10366879403591156, -0.46152403950691223, 0.10077155381441116, 0.1284395158290863, 0.20993514358997345, -0.5283665060997009, 0.2428821176290512, -0.6245034337043762, -0.041666287928819656, 0.2867985963821411, -0.030158983543515205, -0.06601233780384064, -0.6629855036735535, 0.363036185503006, 0.4538419246673584, 0.364574670791626, -0.4009949564933777, 0.36278417706489563, -0.2403123825788498, -0.4649355113506317, 0.6229732036590576, -0.3447430729866028, -0.28659749031066895, 0.3425518870353699, -0.6821503639221191, 0.45194342732429504, 0.04932712763547897, -0.32509690523147583, 0.07591205835342407, 0.07884012907743454, -0.1803646981716156, 0.5617974996566772, 0.3095363676548004, 0.27702468633651733, -0.3225735127925873, 0.35670003294944763, 0.46440935134887695, 0.4841139614582062, -0.5520074963569641, -0.06780154258012772, -0.40782925486564636, 0.5088171362876892, 0.595309853553772, -0.5192633271217346, -0.2665974795818329, -0.2547490894794464, -0.4580641984939575, -0.055593401193618774, 0.37858185172080994};
float b2_c2_bias[8] = {0.03772243484854698, 0.040671207010746, 0.07323469966650009, 0.07508297264575958, 0.09697619825601578, -0.03398921340703964, 0.009675426408648491, 0.06481479108333588};
Block block2(4, // kernel_size
             8, 8, // in_d, out_d
             b2_c1_kernel, b2_c1_bias, b2_c2_kernel, b2_c2_bias);

float layer0_cache_buffer[4*4*4];
RollingCache layer0_cache(
  4, // depth
  4, // dilation
  4, // kernel size
  layer0_cache_buffer
);

float layer1_cache_buffer[8*16*4];
RollingCache layer1_cache(
  8, // depth
  16, // dilation
  4, // kernel size
  layer1_cache_buffer
);

float classifier_weights[8*2] = {0.26889148354530334, 0.024264369159936905, -0.10870025306940079, 0.3449588716030121, -0.5258246660232544, -0.23211383819580078, -0.7355398535728455, 0.434769868850708, 0.564256489276886, -0.5344828367233276, -0.5154579281806946, 0.5851086378097534, 0.8087823987007141, -0.5134549736976624, -0.7756032347679138, -0.09559721499681473};
float classifier_biases[2] = {-0.038666535168886185, 0.09235627949237823};
Classifier classifier(
  8, // input_dim
  2, // output_dim
  classifier_weights,
  classifier_biases
);

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

bool foo = true;
long inference_calls = 0;

void RunInference(float* next_inputs) {

  left_shift_input_buffer.Add(next_inputs);
  block0.Run();
  layer0_cache.Run();
  block1.Run();
  layer1_cache.Run();
  block2.Run();
  classifier.Run();

  inference_calls++;

}

void AudioCallback(AudioHandle::InputBuffer in,
                   AudioHandle::OutputBuffer out,
                   size_t size) {

  cpu_load_meter.OnBlockStart();

  float next_inputs[2];
  next_inputs[0] = 0.4572155;  // one of the common values from training
  float* classifier_out = classifier.GetOutputBuffer();
  for (size_t b = 0; b < size; b++) {
    next_inputs[1] = in[0][b];
    RunInference(next_inputs);
    out[0][b] = classifier_out[0];
    out[1][b] = classifier_out[1];
  }

  cpu_load_meter.OnBlockEnd();
}

void UpdateDisplay() {

  FixedCapStr<100> str("cpu ");
  const float cpu = cpu_load_meter.GetAvgCpuLoad();
  str.AppendFloat(cpu, 5);
  hw.seed.PrintLine(str);

  if (assert_failed) {
    hw.seed.PrintLine(assert_failed_msg);
  } else {
    hw.seed.PrintLine("LGTM 3");
  }

  //RunInference();
  //Write2DArray("classifier.in", classifier.GetInputBuffer(), 1, 8);
  //Write2DArray("classifier.out", classifier.GetOutputBuffer(), 1, 2);

  hw.seed.DelayMs(10);  // ms
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
  AssertSame("b1->l2",
    block1.GetOutputBufferSize(),
    layer1_cache.GetInputBufferSize()
  );
  AssertSame("l1->b2",
    layer1_cache.GetOutputBufferSize(),
    block2.GetInputBufferSize()
  );
  AssertSame("b2->c",
    block2.GetOutputBufferSize(),
    classifier.GetInputBufferSize()
  );

  // connect steps
  left_shift_input_buffer.SetOutputBuffer(block0.GetInputBuffer());
  block0.SetOutputBuffer(layer0_cache.GetInputBuffer());
  layer0_cache.SetOutputBuffer(block1.GetInputBuffer());
  block1.SetOutputBuffer(layer1_cache.GetInputBuffer());
  layer1_cache.SetOutputBuffer(block2.GetInputBuffer());
  block2.SetOutputBuffer(classifier.GetInputBuffer());

  hw.Init();
  hw.SetAudioBlockSize(64); // number of samples handled per callback
  hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);
  hw.StartAdc();

  hw.seed.StartLog();
  cpu_load_meter.Init(hw.AudioSampleRate(), hw.AudioBlockSize());

  hw.StartAudio(AudioCallback);

  while(1) {
    hw.ProcessAllControls();
    UpdateDisplay();
  }

}

