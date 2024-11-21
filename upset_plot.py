from upsetplot import UpSet
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import addcopyfighandler

# Define all 8 sets
sets = {
    "r/opiatesrecovery":{
        "qui9", "Dirty_D_Damnit", "blueskyn01se", "toadlipz85", "misdiagnosisxx1",
        "skunky16", "chasingd0pamine", "The_Motel_", "imagineNimmodium", "bagzplz",
        "chef_710", "oh_suh_dude", "Lipstickcigarette", "callof_thevoid", "ruhicuziam",
        "rez667", "yourmomsfreetime", "Readytoquitforgood", "GettinDrunkWithJesus",
        "nowayjesus1", "telemarketeraddict", "percbuster", "NickyWithdrawl", "vanman33",
        "Gambit2299", "slick718", "themanjb92", "DobusPR", "coffeencigs", "jayisnotathome",
        "Shawshenk", "alividlife", "dori_88", "golf-lip", "SnowboardMore88",
        "piicklechiick", "ChazRhineholdt", "g00seisl00se7", "RiddenHorse", "katsNeo",
        "traceyh415", "kratommd", "BurtSquirtzle", "1Darkgirl", "dirty30blue", "mazaherh",
        "Shaundogg83", "beautifulfuckingmess", "HomanSquareGreenLine", "PhilaDopephia"
    },
    "r/suboxone":{
        "DrSubs", "chasingd0pamine", "Hello_to_u2", "gregsterb", "morganevans73",
        "rplate81", "tristam30", "Jonlaw510", "InfluenceEffective67",
        "Excellent-Attempt204", "thesmallestcock69", "Finallyclean0007", "dodgeorram",
        "airmankenyon", "DarkAngel386", "RC7806", "twisted_guru", "professorpounds420",
        "Nicoledploudre", "bubba2260", "Gabemann2000", "kopitesubuser", "qui9",
        "cruxtone", "HappyThrillmore87", "hambylw_", "burf2500", "Lucid-Design",
        "dnewman3231", "imsaneinthebrain", "FeathercockMelee", "Real-Material344",
        "unbitious", "QuietTHINGno1KNOWS", "StevenRabbi", "TheDarkSideSpoon",
        "Bambino_sharknado", "stronghands_528", "alullintraffic", "takeiteztoma",
        "jax2bcn", "AccountforPills", "DasLIVES88", "iscream80", "kvothethearcane88",
        "ib00sti", "Majestic-Orange", "RecoveryAltONE", "mrowen79", "nampride"
    },
    "r/methadone": {
        "QuietTHINGno1KNOWS", "bg21612121", "mykalASHE", "Excellent-Attempt204",
        "rocksannne", "FeathercockMelee", "OopsNameNotFound", "Shaundogg83",
        "Jaunt-Nominal", "Syrian_lambchop", "imabeerye", "lovelydisputes",
        "stupidusername189", "bree78911", "parabola777", "bagzplz", "nothingt0say",
        "nohopewhytry", "beefbiber", "bonesbrigade619", "ThinkSpinach8819",
        "Layne_Cobain", "junkie8throwaway", "micheleghoulgirl80", "enjoymeredith",
        "Nursefu-qu", "somnifacientsawyer", "EazieWeezie", "dyelyn666", "DARBSTAR_",
        "M367_euphoria", "awall89420", "Yeaheyewilldothat", "Arguswest", "cdurk91",
        "PSThrowaway3", "Jblack401", "Real-Material344", "psychedelic_jesus420",
        "Desperate_timess", "mike9949", "Former-Ad-7561", "Donelifer",
        "AmoBishopRoden83", "betterlatethaneverr", "kenshinmoe",
        "WhatAmiDoingHere1022", "psychedelictravelrr", "Texshroom", "grby1900"
    },
    "r/addiction": {
        "Babyyy_Kattt", "alphatweaker", "IrelynneGemini", "sarabeth314", "xluzix",
        "m_rea", "j_p420", "MaJFn", "Upstate_NY518", "yourmomsfreetime",
        "largecucumber", "skillsforilz", "pixiemajik5", "allisonovo", "iloveluckie",
        "waismannmethod", "the_lone_researcher", "ard2424", "ArtemisJohn",
        "HopeThisHelps90", "psychguyjeff", "pillhead123", "KayleeBabby2020",
        "fu11m3ta1", "Singngkiltmygrandma", "Recovering_Addict_LT", "andreaidkk",
        "JLD58", "SoSoberMom", "NeverAloneRecovery", "blueberrybird24",
        "gosmurfyourself69", "p5ycliqu3", "loudbounce", "ItsToxii", "cragpossum",
        "Jaunt-Nominal", "KAYO_STL_MO", "spacesmitten", "Few_Lifeguard_5220",
        "Real-Material344", "TaurineLine719", "cigarrette", "pagex",
        "TracBExchangeLV", "drugaddict30", "melmuth", "yoitsjustin", "usuario_unico", "Icy_Issue499"
    },
    "r/opiates": {
        "traceyh415", "DankRecovery", "morbo2000", "MetroMaker", "spinderella69",
        "StoneyGwynn", "FashionablyFake", "KickerS12X", "randonme", "dori_88",
        "FrmBURGHinCHI", "DopeHammahead", "berryfrezh", "travs3dpe", "TATP1982",
        "chasingd0pamine", "Mellomelll", "skipper489", "inlovewithheroin", "2ndwaveobserver",
        "TheDarkSideSpoon", "schizoidparanoid", "Dopana", "ASavageLost", "PercSet",
        "thelarustatrust", "DasLIVES88", "Dilly-dallier", "monwymike42314", "slick718",
        "ThatYoungBro", "psychedelicnickk", "jlogic420", "douoweme", "PM_ME_PCP",
        "Z1gg0Z_420", "Gutterlungz1", "OhNoImAnOreo", "TheEater_OfDreams", "NoseCandiez",
        "flatline904", "Shaou_Lin", "OlDirtyBurton", "412dopefool", "dyingsober",
        "DawgfoodMN", "northwest_vae", "moon_meander", "Oxiconone", "UsamaBinNoddin"
    },
    "r/heroin": {
        "xluzix", "CeltiaDogy", "BlueXanzCan", "MetroMaker", "CHIskyhigh", "jwcoffee",
        "Commonwealthkyle9000", "InLovewithH", "whatdrugswhat", "poopypoppies", "LaxerFL420",
        "skinnypetitethang", "midwest_dope23", "shade1994", "666yungxchrist59", "cshallll95",
        "barely_there_atall", "mathildabrbdg", "G98Ahzrukal", "DipsburghPa", "MonkeyCultLeader",
        "DegenerateDrugUse", "randaniicole", "staggerlee4242", "yaboyirish", "LimitedPastabilities",
        "willatkins408", "JunkieJeezus", "cman_the_bartard", "ZealousidealSun3226", "Don1994oneill",
        "PyschoActiveGoon", "Duke54327", "drugy710", "Codeinebabyk", "throwaway3153100313",
        "dope_scramble", "Voodoo_Gumpthrie", "Lawson98_001", "woodspleasedream", "jahcarter2020",
        "2cool2fresh", "Luck-Spell", "unrulyhair", "iheartoxycotin_1998", "Gorilla69420",
        "EndNaive", "cam32596", "xCancerberox", "richmanshigh"
    },
    "r/opiatechurch": {
        "Oxy15mg", "Ajm6753", "Lil_Roxi2", "Hugheydee", "dyingoutwest94", "snicktheboss",
        "adderalandphenibut", "DrMrPoppy", "1UpTahpAhk", "Oxykodeen", "brooksrobinson83",
        "Au79ine", "eastcoastbitch", "ronaldobeezley", "2cups_", "matrixman89",
        "Disposable187", "fobandkill", "perc30nowitzki", "jrodriguez1091", "Tiffytyler13",
        "tankpem", "btakeover", "OxyCock209", "Lil-bigswit", "XCerealKLLrx420",
        "anxiousghost3", "Oxy20", "CalebSt_21", "Aqua2blue", "panda_nips",
        "lofent1690", "PercG0D1135", "yermawsmokeschips", "urwomansfav", "roxicoedone",
        "oxygod30", "RoxiBalboa", "theheynowman", "unalert", "imtheluckyloser",
        "TheycallmePrez", "Big_Honeydew_7456", "Neithman1996", "nash-got-hash", "solodomo20",
        "SayNo2Drux", "slump30mg", "hotdamnisme", "gassed_up_shawtyyy"
    }, 
        "r/poppytea":{
        "somniferumphile", "adamole123", "forcetohaveaname", "mrfuzzyasshole", "tussinNEXT",
        "FritzItzig", "PODrickPAYNEless", "dr4g0n6t00", "JohnJoint", "_Hypnos_",
        "PST_Jim", "CharlieLemon", "kzrsosa", "Luvsmepst", "iluvopies", "grizzythekid",
        "pop-n-lox", "Psychonaut424", "Tjmaxwell12", "khaoticrumpus", "Teyak84",
        "monkeyjorts", "HarpuaUnbound", "Palmer1997", "Jdm374", "Bfw1977", "Wvzombie138",
        "Roxy1131", "splitchin", "TomBosley12", "officialsoulresin", "RIPHenchman24",
        "TheeJimmyHoffa", "BulletProofSnork", "Akaryrye", "FastBreakz", "PoppyGirl99",
        "payday_vacay", "Emilio_Estevez_", "Whathappened4513", "notalltogetherhere",
        "ChromeBitchSickTrips", "DaturaCurtains", "ethylnaut", "shamanskat", "bondageman420",
        "Apprehensive_Boat247", "JLeww69", "Sbchick887", "ercocet"
    }
}
for item in sets:
    print(item, len(sets[item]))
# Prepare the Counter data for overlaps
all_users = set.union(*sets.values())
counter = Counter(
    tuple(user in sets[file] for file in sets.keys()) for user in all_users
)

# # Convert Counter to DataFrame with MultiIndex
# df = pd.DataFrame.from_dict(counter, orient="index", columns=["count"])
# df.index = pd.MultiIndex.from_tuples(df.index, names=list(sets.keys()))

# # Plot the UpSet plot
# upset = UpSet(df["count"])
# upset.plot()
# plt.title("Overlaps Between All 8 Files")
# plt.show()
df = pd.DataFrame.from_dict(counter, orient="index", columns=["count"])
df.index = pd.MultiIndex.from_tuples(df.index, names=list(sets.keys()))

# Plot the UpSet plot
upset = UpSet(df["count"], intersection_plot_elements=10)
plot = upset.plot()

# Add annotations (numbers) on top of the bars
ax = plot['intersections']
for bar in ax.patches:
    height = bar.get_height()
    if height > 0:  # Only annotate bars with height > 0
        ax.annotate(f'{int(height)}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height), 
                    xytext=(0, 5),  # Offset text above bar
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.title("Overlaps Between All 8 Files")
plt.show()