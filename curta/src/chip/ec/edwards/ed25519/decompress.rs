use itertools::Itertools;

use super::params::{Ed25519BaseField, Ed25519Parameters};
use super::point::CompressedPointRegister;
use super::sqrt::Ed25519FpSqrtInstruction;
use crate::chip::builder::AirBuilder;
use crate::chip::ec::edwards::{EdwardsCurve, EdwardsParameters};
use crate::chip::ec::point::AffinePointRegister;
use crate::chip::field::instruction::FromFieldInstruction;
use crate::chip::field::parameters::FieldParameters;
use crate::chip::field::register::FieldRegister;
use crate::chip::AirParameters;
use crate::math::field::Field;
use crate::polynomial::Polynomial;

impl<L: AirParameters> AirBuilder<L> {
    pub fn ed25519_decompress(
        &mut self,
        compressed_p: &CompressedPointRegister,
    ) -> AffinePointRegister<EdwardsCurve<Ed25519Parameters>>
    where
        L::Instruction: FromFieldInstruction<Ed25519BaseField> + From<Ed25519FpSqrtInstruction>,
    {
        // Ed25519 Elliptic Curve Decompress Formula
        //
        // This function uses a similar logic as this function:
        // https://github.com/succinctlabs/curve25519-dalek/blob/e2d1bd10d6d772af07cac5c8161cd7655016af6d/curve25519-dalek/src/edwards.rs#L187
        let num_limbs: usize = Ed25519BaseField::NB_LIMBS;

        let mut one_limbs = vec![0u16; num_limbs];
        one_limbs[0] = 1;
        let one_p = Polynomial::<L::Field>::from_coefficients(
            one_limbs
                .iter()
                .map(|x| L::Field::from_canonical_u16(*x))
                .collect_vec(),
        );
        let one = self.constant(&one_p);

        let d_p = Polynomial::<L::Field>::from_coefficients(
            Ed25519Parameters::D[0..num_limbs]
                .iter()
                .map(|x| L::Field::from_canonical_u16(*x))
                .collect_vec(),
        );
        let d: FieldRegister<Ed25519BaseField> = self.constant(&d_p);

        let zero_limbs = vec![0; num_limbs];
        let zero_p = Polynomial::<L::Field>::from_coefficients(
            zero_limbs
                .iter()
                .map(|x| L::Field::from_canonical_u16(*x))
                .collect_vec(),
        );
        let zero: FieldRegister<Ed25519BaseField> = self.constant(&zero_p);

        let yy = self.fp_mul::<Ed25519BaseField>(&compressed_p.y, &compressed_p.y);
        let u = self.fp_sub::<Ed25519BaseField>(&yy, &one);
        let dyy = self.fp_mul::<Ed25519BaseField>(&d, &yy);
        let v = self.fp_add::<Ed25519BaseField>(&one, &dyy);
        let u_div_v = self.fp_div::<Ed25519BaseField>(&u, &v);

        let mut x = self.ed25519_sqrt(&u_div_v);
        let neg_x = self.fp_sub::<Ed25519BaseField>(&zero, &x);
        x = self.select(&compressed_p.sign, &neg_x, &x);

        AffinePointRegister::<EdwardsCurve<Ed25519Parameters>>::new(x, compressed_p.y)
    }
}

#[cfg(test)]
mod tests {
    use core::str::FromStr;

    use curve25519_dalek::edwards::CompressedEdwardsY;
    use num::BigUint;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::chip::builder::tests::*;
    use crate::chip::ec::edwards::ed25519::gadget::{CompressedPointGadget, CompressedPointWriter};
    use crate::chip::ec::edwards::ed25519::instruction::Ed25519FpInstruction;
    use crate::chip::ec::gadget::{EllipticCurveGadget, EllipticCurveWriter};
    use crate::chip::ec::point::AffinePoint;

    #[derive(Clone, Debug, Copy, Serialize, Deserialize)]
    pub struct Ed25519DecompressTest;

    impl AirParameters for Ed25519DecompressTest {
        type Field = GoldilocksField;
        type CubicParams = GoldilocksCubicParameters;

        const NUM_ARITHMETIC_COLUMNS: usize = 816;
        const NUM_FREE_COLUMNS: usize = 3;
        const EXTENDED_COLUMNS: usize = 1233;
        type Instruction = Ed25519FpInstruction;
    }

    const NUM_TEST_CASES: usize = 51;

    const COMPRESSED_P: [&str; NUM_TEST_CASES] = [
        "02f80695f0a4a2308246c88134b2de759e347d527189742dd42e98724bd5a9bc",
        "04d3b737505bfbf1eee1375118f8d584302f2da7eefaa3c5d7b095f3cb485938",
        "064a31abd0d2431f3109a44c8b00e724f56731a99852408c83985157fa6276da",
        "092005a6f7a58a98df5f9b8d186b9877f12b603aa06c7debf0f610d5a49f9ed7",
        "0a978fd659c69448273e35554e21bac35458fe2b199f8b8fb81a6488ee99c734",
        "14574d49c457d877e91db73a93ac8ca5fc595ca25c25f156398f32322dd71f59",
        "1e05e4b40cf57ae8965cac3ea994a135f020b6a4e02ac478d3025dfe2f33d12c",
        "244f13c0835db4a3909cee6106b276684aba0f8d8b1b0ba02dff4d659b081adf",
        "262b5e095b309af2b0eae1c554e03b6cc4a5a0df207b662b329623f27fdce8d0",
        "290d9479d0c1998373900dfb72900d1c9f55c018dd7eeed4ce0e988bb3da03a2",
        "2f4515da05794e97d8a4b15679913ca844fa4e25cbe936aa92afce02ceac8071",
        "301d8f635ec49e983cc537c4b81399bb24027ac4be709ce1a4eeb448e98a9aec",
        "3ed9fc282b5a8c2203d79501f4d1ba9b673196c6684b17bba0e3380637ddf578",
        "496f13c8d934c49cf357ff744fe0adb9ba8bb4c0a6697194ab6c807be92f590a",
        "53bdf319c8bf5992e7ad6215aa2dda2db44e6cc0caad127544a0f83362bca022",
        "597f675a8d47cb4165e9eede08ef9f7cb5906cbc9b44da080a21f9f9e905f730",
        "5b8afe90b542f6e0fffb744f1dba74e34bb4d3ea6c84e49796f5e549781a2f5c",
        "5da3581ce065c8037217ecf718fc9b97e40bd40f36038f1136135c9ee5a9d5f2",
        "61771bd363e7deb0e9f260e9a7e0e36e646cdb30ae5e8b7ed55ce45411a4ac4c",
        "69cfb82e998e36468e79b3eaeec213d9ec9d76e25edc1b65d41c85af32cca365",
        "6d5e025e08e3ec72531c476098287b13295fa606fab8275418e0c4c54f236c9e",
        "6db5e3b16a2cc4fc1de701c21a073981eb1a50caeb2e0294a21b786e92f08475",
        "73071024c4dfbc81863e7201db88fc1a6fa6dcc4f6f2f0a8a33ef8ff3ef94473",
        "73fbfdaa00a5205310cb0d1bd54175647482fae300cc66b36e7846e82288e9f0",
        "7584413f927bee5a607d36d9eedc451ad5dab66af39d53c548649260e0676014",
        "8207bf4bf7e95006f8da5463b44c77c8e334456e2c4c4d4cd5ffb74e8e45d071",
        "840bd86fa97ef694a34fd47740c2d44ff7378d773ee090903796a719697e67d8",
        "88554d82b1e63bedeb3fe9bd7754c7deccdfe277bcbfad4bbaff6302d3488bd2",
        "88ae88472391e18ec1b416f69ccf13399c4a0f6cc6abe662a88e033fb3f74595",
        "8c9b6893834f9af8cf8410ed26f6e9c8029891b9b08e36d2706226d420580423",
        "94028d6eafa1c0d6218077d174cc59cea6f2ea17ef1c002160e549f43b03112b",
        "95a90ca29491de8982145611188f1a18bba22e29371d0ab494cd8bd0f02ab5b5",
        "95fefa2edb64720ba97bd4d5f82614236b3a1f5deb344df02d095fccfe1db9b0",
        "964906e479f2172f4eaea2446e44964680213193ac981c3ca77015df8b02f4c8",
        "978a06ff17956afd5cbb8a1e85267dccad9ea0aef5cb75f1c22af1baf2901358",
        "a1b99a0a349691e80429667831f9b818431514bb2763e26e94a65428d22f3827",
        "a8565f4f6e753fc7942fa29051e258da2e06d13b352220b9eadb31d8ead7f88b",
        "a9e1f3861a27d63510f7ff632506216abd09c1f9c0233c558bda127ef07f018e",
        "ad19893bd78c611087f21c0f33ca1edbb053aa5fdb3aa00ff6a4ba7d6ba286b8",
        "c3322124f3e4161dd1b57a7bb412c9352084215b1ceb15550850b3452a32d467",
        "c39ad005c6ce056b0fa7baff65f4d77ed645aef3488080396c568f1c7989c60a",
        "c4a1b230780455f5f18ec4b84386cde822e62ceefa8d32fbca0b9f398e434c86",
        "dd66c7a479fe0352fda60876f173519b4e567f0a0f0798d25e198603c1c5569b",
        "dea0d85a4cf74a91c85d7f90873bfbdc40c8c939377bec9a26d66b895a1bbeaa",
        "df4bc26d8aeb83ed380c04fe8aa4f23678989ebffd29c647eb96d4999b4a6736",
        "e2eefb4dd06c686ca29cdb2173a53ec8322a6cb9128e3b7cdf4bf5a5c2e8906b",
        "eab6f3a26d7fd65eff7c72a539dbeee68a9497476b69082958eae7d6a7f0f1d5",
        "f86c72277ed20efe15bab1abcf34656e7d2336e42133fa99331e874b5458b28f",
        "fa74875360c9fab71916824e96bccf6417c4ea23f01c298f31bc8775689dd93b",
        "fc8f03e8f87dc94bcb10b18b8ad3d2394533c41b51c0fc6f0c89d7498bc07cf2",
        "fe249696419a67cb9e0f29d0297d840048bddf6612a383f37d7b96348a1bc5f1",
    ];
    const X_VALUES: [&str; NUM_TEST_CASES] = [
        "42944753775096595453317774301569232845918873255423446399126847946286666042083",
        "2617378013254491276526480142297006654950083246743655934168645664790954172856",
        "40554179325058007786825212834019412451821159259968020232872927078948867472777",
        "46498302853928694666678954892487633213173616891947474062122140763021808627271",
        "7046554187569412910872426625416169129199735897936767053104166896760954666796",
        "51836682789722212998070589808486703504866185344970744025068123110096985515182",
        "9936419860705630055026733279082032353780628122094614068231006743972146476618",
        "20355506797543641690028825599103494750539470816527716641873844937549362986955",
        "34442166386923719205699015534895825347504009186315300478087979683688844882877",
        "31786027486947645354498946468334706334451286370447941064032551733539978498135",
        "13498918434099771043375368119295426241908587538498624849259990044763558675598",
        "42588447688499296919973420948400443336047911072100500921339156079009697054209",
        "49023995333124755187077000337465801345488316013828460963889956264045903141772",
        "44678895033842239414209168581003001126046613876786803401795394113177733619116",
        "5997798642223075884184846431088773174606425858337680801206306097925859638778",
        "36848274817610251565124028119310850793797686607251119363504793748076306044766",
        "38472758985697376631116932049808603691497227691421161449407163800226749648854",
        "52212373723952553740876809590858635859683421505139903371738506313403071198885",
        "38282403937854902166127332021293082012658465433377563783302969384475425041182",
        "32814888498853882657994494035842816674263144794889413992085702791635794345074",
        "7775748190622831573130569765402390670918205913959641024445216629007713224615",
        "12269865905511380530953553714818547398303136090338119901588978942084234647832",
        "27296074514809263717463796322131251960062692657123382359592358885552252979566",
        "29511526243536081678046628975839351493714236510876614216824379641822201968713",
        "2141927634644650956447128733561017256355504247996330933110439208079201853304",
        "9072925988495429431969231841284558183142588467005755949443824311557530190026",
        "46060523643801503732002440450077403853321923367696886059174080860391713325141",
        "51775433507790414213913312315404518046894067500571146935607299928340826995145",
        "43577208670791693779893588139968137442831384413787117477341475331111719269017",
        "52628458098862889101770235450346115053070262648443390636091886727938227964784",
        "5398902269331041183731924084567619944618412426872820745569859849049604873208",
        "6364386142667837393491097861455946906731657923492378560737482646505723598147",
        "56032943743058340413777052656252618265263894351250898233581079636761870394245",
        "52539404658252790688455943898902738438323861028117544686903595483558495071859",
        "4971005134919009505155667266609648750747474136651618305072438337858037356016",
        "4571747608125990812334587385891647722075299974882740846499616421772624102100",
        "4853853789621584269695621636434295378543916985457815017054572321397150956985",
        "43972769770055304876363347295266542494674938890560391383802917358718117817339",
        "25647206415972931168560974513406056430871753746932372288566481198197491489949",
        "4815982148420782874976346500223036751578828598978743069037545646966904541450",
        "22883363510129566194975596629770116998307605337710581184274103181434929929254",
        "18508868780790789143684476680485791129976701723136044142314973479101617595293",
        "36738011564783376991103385894330612539938402592664564816943185807740479358031",
        "46310166708701398919036870881896588351018298224004158746306249155741571819651",
        "8912291264024987854331665004024267727972251797701020265912536295022306500858",
        "50255019135077322510719518023799219399736988833379566248313155058433904600384",
        "48424894585296919372832557658888315879819741270100832142610560272116014743533",
        "35974974659372589827444172322478881263810188699200584399585152163197701969617",
        "33760042870555248298765414973292585013305521168889657131791144891817665858562",
        "47192355000571364234585660354516864531790745370096032962516566788742182473669",
        "15366489060607989673625663665088391596910013188155705585133409798882026774993",
    ];
    const Y_VALUES: [&str; NUM_TEST_CASES] = [
        "27438840174979602354887622826255875593675698506531955421768075258066108872706",
        "25487271333686802402150046179160400030714984463072321363297370904298115289860",
        "40917327446462212490173356058941946423622517572361693502134168982445312002566",
        "39631481484518050587569957685312118971016858466473658974116614941450822819849",
        "23872933090242768634864081646065013449130967826229958668051150342040488744714",
        "40312100876867412014897335132093129918326394879258068759871344826948590065428",
        "20271389657171167694219380445464084850658735042111796269059130372583344833822",
        "43015718042525680097742254293397816450594350490653493214746153148159414718244",
        "36596458239299630990787045415025381210384855327554228659505180133585019874086",
        "15385446814260407631096255468392313780299757992460080175561700844303484718377",
        "51338700968617046693202931351473218181548069252191836096758635084946527831343",
        "49122840825288987866423333188249007964307746201960468668557580359730887335216",
        "54711946130257654771102641190247083628434804336313385478426207598713748904254",
        "4680708551350055339063038169074304953492514389805396873155582858730654625609",
        "15662632558056774401632038132291426973190892213441369081786010203488295042387",
        "22147468773721601346352171317862045808089007795051923518956756746674628951897",
        "41696006570086105479393300653642830981999578439380751383247904176358641535579",
        "51941175748927778079204080618220084756921304540275579880901913614960672940893",
        "34680806541130270053434693034931489503372390466105470800728554847114773624673",
        "45973003101205270899990870352828733533053923410878744484590985424018096050025",
        "13760448652279492822278164142383839202177309496020642255450854279257945955949",
        "53155487463702118938594077776520666890738206112578951211505915906389082092909",
        "52137843420794365326571431834058654340244220322101287719192084373630975215475",
        "51071653986021990770705862010348446931181051888141401665693243720548849941363",
        "9216591218956027600037799201403073005141407587384494292487319037444064642165",
        "51479336136483412802770919453660573483538588491958522088373635147566178830210",
        "39986388376535459921128071358773226664034210674748635681634884053437750184836",
        "37335747940369516717531011006848410450597473068334258258963037747894285063560",
        "9622191831532551109518591409888394459734930760506720589639918391937223667336",
        "15838625327412244363965263842785254981685539807429087352656551706846942501772",
        "19479511210784179184757101076824296309435647410846257555950558995082231218836",
        "24292676659334239773424872889123952603285034555966670829131492202601429903765",
        "22038090458974101109931012144507576982190310692666359590057510913308597354133",
        "32997653356219847022189240942543229222088798821904817061835501015543809722774",
        "39838101165026559385761250085576019260518414333301950746206584509431095397015",
        "17739474578289392873039985595276850249192918758238202354754192635935908870561",
        "5415109613341015494355534457744649790009669669988738624054174646725863495336",
        "6335029732680855047157571274942240493793361464925569651571127366584762622377",
        "25567398008206512208396852999506744845799094052417996212983689139484831062445",
        "46963141208784868090257508439768598498935617657633463209198842223007054770883",
        "4873913009069550844109600156486449642191904579124543130450335743465214745283",
        "2848623719808645624166543783296899495191700988091694528221553665220214432196",
        "12365760606990260085901868857389550828780325344868287997506478478472147068637",
        "19333029370823790055651017141821284640159680619771947318440737070092553593054",
        "24607393995390189352588738065111674380392924946661520643648903036192692521951",
        "48653507228597136940069526486214913450481583891894698410588749203406518480610",
        "38874063216255791666777928049216569098111646865132339476809688504309925525226",
        "7099801132547592032533135806989275475620225580037696611842802107721182244088",
        "27070950269857902135198241826954301918534387061658993701917689718619213624570",
        "51784082665032521843957296683250959413254977647991560658398692887954469392380",
        "51459610834832591042776722808167475162304165568946445662728688763073384031486",
    ];

    #[test]
    fn test_ed25519_decompress() {
        type L = Ed25519DecompressTest;
        type SC = PoseidonGoldilocksStarkConfig;

        let mut builder = AirBuilder::<L>::new();

        let compressed_p_reg = builder.alloc_ec_compressed_point();
        let affine_p_reg = builder.ed25519_decompress(&compressed_p_reg);
        let expected_affine_p = builder.alloc_ec_point();
        builder.assert_equal(&expected_affine_p.x, &affine_p_reg.x);
        builder.assert_equal(&expected_affine_p.y, &affine_p_reg.y);

        let num_rows = 1 << 16;
        let (air, trace_data) = builder.build();
        let generator = ArithmeticGenerator::<L>::new(trace_data, num_rows);

        let writer = generator.new_writer();
        writer.write_global_instructions(&generator.air_data);

        (0..num_rows).into_par_iter().for_each(|i| {
            let compressed_p_bytes = hex::decode(COMPRESSED_P[i % NUM_TEST_CASES]).unwrap();
            let compressed_p = CompressedEdwardsY(compressed_p_bytes.try_into().unwrap());

            let affine_p_x = BigUint::from_str(X_VALUES[i % NUM_TEST_CASES]).unwrap();
            let affine_p_y = BigUint::from_str(Y_VALUES[i % NUM_TEST_CASES]).unwrap();
            let affine_p = AffinePoint::<EdwardsCurve<Ed25519Parameters>>::new(
                affine_p_x.clone(),
                affine_p_y.clone(),
            );

            writer.write_ec_compressed_point(&compressed_p_reg, &compressed_p, i);
            writer.write_ec_point(&expected_affine_p, &affine_p, i);
            writer.write_row_instructions(&generator.air_data, i);
        });

        let stark = Starky::new(air);
        let config = SC::standard_fast_config(num_rows);
        let public = writer.public().unwrap().clone();

        // Generate proof and verify as a stark
        test_starky(&stark, &config, &generator, &public);

        // Test the recursive proof.
        test_recursive_starky(stark, config, generator, &public);
    }
}
