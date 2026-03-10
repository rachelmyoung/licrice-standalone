"""Testing utility functions for LICRICE.
Extracted from pyTC.testing
"""

from licrice.tracks import utils as tutils


def trackset_integrity_check(trackset, test_var, basevars):
    """Test variable missingness."""
    for bv in basevars:
        # check non-missingness and greater than 1 requirements are met
        g2g, one_ol_ob, missing = tutils.assess_var_missingness(
            trackset,
            test_var,
            bv,
            0,
        )
        if g2g.storm.shape != trackset.storm.shape:
            assert len(one_ol_ob.storm) == 0, (
                f"There are storms with one or less {test_var} observation."
            )
            assert len(missing.storm) == 0, (
                f"There are storms with missing {test_var} observations."
            )


def boolean_array_check(boolean_array, ds):
    """Check if a test holds true for all storms, if it do."""
    assert boolean_array.all(), print(
        "For {} test, these storms failed: {}".format(
            boolean_array.name,
            ds[{"storm": ~boolean_array}].storm,
        ),
    )
