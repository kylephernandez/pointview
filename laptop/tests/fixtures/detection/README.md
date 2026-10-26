# Detection fixtures

## `street_porlamar.jpg`

960 x 557 JPEG (190 KB): a street in Porlamar, Venezuela, with people, parked cars, an SUV and a
motorcycle. Used by `tests/detection/test_real_model.py` (`@pytest.mark.model`).

* Title: "Traffic and pedestrians on Calle Velasquez near Catedral San Nicolas de Bari in Porlamar, Venezuela"
* Source: https://commons.wikimedia.org/wiki/File:Traffic_and_pedestrians_on_Calle_Vel%C3%A1squez_near_Catedral_San_Nicol%C3%A1s_de_Bari_in_Porlamar,_Venezuela.jpg
* License: CC0 1.0 Universal (public domain dedication), per the Wikimedia Commons file page
  (`LicenseShortName` = "CC0" in the Commons API, 2026-10-26).
* Modification: Commons' 960 px-wide JPEG thumbnail, otherwise unchanged.

Expected by the test (real model, score >= 0.5): at least 4 people, at least 2 cars/trucks, 1 motorcycle.
The photo shows identifiable passers-by and a license plate; it is a test fixture only. PointView itself
never labels who a person is.
