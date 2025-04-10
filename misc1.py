

from histocc import OccCANINE
import pandas as pd

occ1950_desc = pd.read_csv(r"D:\Dropbox\Research_projects\HISCO\OccCANINE\Data\OCC1950_definitions.csv")

mod_icem = OccCANINE(r"D:\Dropbox\Research_projects\HISCO\OccCANINE\Data\models\mixer-icem-ft\last.bin", hf = False, system="icem")
mod_isco = OccCANINE(r"D:\Dropbox\Research_projects\HISCO\OccCANINE\Data\models\mixer-isco-ft\last.bin", hf = False, system="isco")
mod_occ1950 = OccCANINE(r"D:\Dropbox\Research_projects\HISCO\OccCANINE\Data\models\mixer-occ1950-ft\last.bin", hf = False, system="occ1950", descriptions=occ1950_desc)
mod_psti = OccCANINE(r"D:\Dropbox\Research_projects\HISCO\OccCANINE\Data\models\mixer-psti-ft\last.bin", hf = False, system="psti", use_within_block_sep=True)
mod = OccCANINE()

print(mod_icem(["he is a farmer", "mason", "finest taylor"], lang = "en"))
print(mod_isco(["he is a farmer", "mason", "finest taylor"], lang = "en"))
print(mod_occ1950(["he is a farmer", "mason", "finest taylor"], lang = "en"))
print(mod_psti(["he is a farmer", "mason", "finest taylor"], lang = "en"))
print(mod(["he is a farmer", "mason", "finest taylor"], lang = "en"))


print(mod_icem(["he is a farmer", "mason", "finest taylor"], lang = "en", prediction_type = "flat"))
print(mod_isco(["he is a farmer", "mason", "finest taylor"], lang = "en", prediction_type = "flat"))
print(mod_occ1950(["he is a farmer", "mason", "finest taylor"], lang = "en", prediction_type = "flat"))
print(mod_psti(["he is a farmer", "mason", "finest taylor"], lang = "en", prediction_type = "flat"))
print(mod(["he is a farmer", "mason", "finest taylor"], lang = "en", prediction_type = "flat"))

print(mod_icem(["he is a farmer", "mason", "finest taylor"], lang = "en", prediction_type = "full"))
print(mod_isco(["he is a farmer", "mason", "finest taylor"], lang = "en", prediction_type = "full"))
print(mod_occ1950(["he is a farmer", "mason", "finest taylor"], lang = "en", prediction_type = "full"))
print(mod_psti(["he is a farmer", "mason", "finest taylor"], lang = "en", prediction_type = "full"))
print(mod(["he is a farmer", "mason", "finest taylor"], lang = "en", prediction_type = "full"))