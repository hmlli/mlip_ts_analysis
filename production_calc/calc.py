def get_calculator(mlip_method):
    if mlip_method == "MACE":
        from mace.calculators import mace_mp
        calc = mace_mp(model="medium-mpa-0", dispersion=False, default_dtype="float32", device='cpu')

    elif mlip_method == "GRACE":
        from tensorpotential.calculator.foundation_models import grace_fm, GRACEModels
        calc = grace_fm(GRACEModels.GRACE_2L_OAM)

    elif mlip_method == "mattersim":
        from mattersim.forcefield import MatterSimCalculator
        device = "cpu"
        calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)

    elif mlip_method == "deepmd":
        from deepmd.calculator import DP
        calc = DP(model="2025-01-10-dpa3-openlam.pth")

    elif mlip_method == "eqV2-M":
        from fairchem.core.models.model_registry import model_name_to_local_file
        checkpoint_path = model_name_to_local_file('EquiformerV2-86M-OMAT24-MP-sAlex', local_cache='/tmp/fairchem_checkpoints/')
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=True)

    elif mlip_method == "sevenn":
        from sevenn.calculator import SevenNetCalculator
        calc = SevenNetCalculator('7net-mf-ompa', modal='mpa')

    elif mlip_method == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator
        calc = CHGNetCalculator()

    return calc
