import numpy as np
import pandas as pd
import typing
import datetime

from ns_sfd import constants
from ns_sfd import epoch
from ns_sfd import orbital_state
from ns_sfd import propagator
from ns_sfd import dynamics
from ns_sfd import datautils
from ns_sfd import utils
from ns_sfd.force.drag.atmosphere import nrlmsis
from ns_sfd.force.drag.atmosphere import nrlmsis00_indices
from ns_sfd.force.drag.dragobject import isotropic_drag

datautils.data.set_data_dir("Notebooks/data/orekit-data")

def convert_keplerian_elements_to_orbital_state(
    kep: orbital_state.KeplerianElements, mass: float
) -> orbital_state.OrbitalState:

    c = kep.cartesian_coordinates

    e = epoch.Epoch.from_datetime(kep.epoch)

    return orbital_state.OrbitalState(
        c[0], c[1], c[2], c[3], c[4], c[5], epoch=e, mass=mass)


def kepler_orbit_from_keplerian_elements(
    kep: orbital_state.KeplerianElements,
    mass: float,
    step: float,
    duration: float
) -> typing.List[orbital_state.OrbitalState]:
    state0 = convert_keplerian_elements_to_orbital_state(kep, mass)

    prop = propagator.KeplerianPropagator(state0)

    ef = state0.get_epoch().get_shifted_epoch(duration)

    return prop.get_states(step, ef)


def full_dynamics_orbit_from_keplerian_elements(
    kep: orbital_state.KeplerianElements,
    mass: float,
    step: float,
    duration: float
) -> typing.List[orbital_state.OrbitalState]:

    state0 = convert_keplerian_elements_to_orbital_state(kep, mass)

    dyn = dynamics.Dynamics()
    dyn.add_gravity(70, 70)

    atm_model = nrlmsis.NRLMSIS00(
        nrlmsis00_indices.NrlmsisIndicesLongTermPrediction()
    )
    drag_satellite = isotropic_drag.IsotropicDrag(
        drag_area=10.0, drag_coefficient=3.0)

    dyn.add_drag(atm_model, drag_satellite)

    prop = propagator.Propagator(state0, dyn.get_force_models())

    ef = state0.get_epoch().get_shifted_epoch(duration)

    return prop.get_states(step, ef)


def states_to_dataframe(
        states: typing.List[orbital_state.OrbitalState]) -> pd.DataFrame:
    
    e0 = states[0].get_epoch()

    dt = [s.get_epoch().get_duration_from(e0) for s in states]

    coordinates = np.array([s.get_coordinates() for s in states])

    acc = np.array(
        [utils._orekit_vector3d_to_array(
            s._get_service_object().pVCoordinates.getAcceleration()
            ) for s in states])


    data = {
        "time": dt,
        "x": coordinates[:, 0],
        "y": coordinates[:, 1],
        "z": coordinates[:, 2],
        "xdot": coordinates[:, 3],
        "ydot": coordinates[:, 4],
        "zdot": coordinates[:, 5],
        "xdotdot": acc[:, 0],
        "ydotdot": acc[:, 1],
        "zdotdot": acc[:, 2],
    }

    return pd.DataFrame(data)


def training_data_from_kep(
    kep: orbital_state.KeplerianElements,
    m: float,
    step: float,
    duration: float
) -> typing.Dict:
    states = kepler_orbit_from_keplerian_elements(kep, m, step, duration)

    df = states_to_dataframe(states)
    training_data = df[["x", "y", "z", "xdot", "ydot", "zdot"]].to_numpy()
    t = df["time"].to_numpy()

    derivative = df[
        ["xdot", "ydot", "zdot", "xdotdot", "ydotdot", "zdotdot"]].to_numpy()


    return {"t": t, "training_data": training_data, "derivative": derivative}

def data_from_kep(
    kep: orbital_state.KeplerianElements,
    m: float,
    step: float,
    duration: float
) -> pd.DataFrame:
    states = kepler_orbit_from_keplerian_elements(kep, m, step, duration)

    return states_to_dataframe(states)


def full_dynamics_training_data_from_kep(
    kep: orbital_state.KeplerianElements,
    m: float,
    step: float,
    duration: float
) -> typing.Dict:
    states = full_dynamics_orbit_from_keplerian_elements(kep, m, step, duration)

    dataframe = states_to_dataframe(states)

    atm_model = nrlmsis.NRLMSIS00(nrlmsis00_indices.NrlmsisIndicesLongTermPrediction())
    drag_satellite = isotropic_drag.IsotropicDrag(drag_area=10.0, drag_coefficient=3.0)

    density_values = [atm_model.get_density(state) for state in states]
    drag_coefficient_value = drag_satellite.get_drag_coefficient()
    drag_area_value = drag_satellite.get_drag_area()

    dataframe['density'] = density_values
    dataframe['drag_coefficient'] = drag_coefficient_value
    dataframe['drag_area'] = drag_area_value

    return dataframe

def full_dynamics_training_data_from_kep_with_orbital_elements(
    kep: orbital_state.KeplerianElements,
    m: float,
    step: float,
    duration: float
) -> typing.Dict:
    states = full_dynamics_orbit_from_keplerian_elements(kep, m, step, duration)

    df = states_to_data_frame_with_orbital_elements(states)
    cartesian_orbit = df[["x", "y", "z", "xdot", "ydot", "zdot"]].to_numpy()
    kep = df[
        ["sma", "ecc", "inc", "arg_perigee", "raan", "true_anomaly"]
    ].to_numpy()
    t = df["time"].to_numpy()

    return {"t": t, "cartesian": cartesian_orbit, "kep": kep}


def osculating_elements_to_cartesian(
    osculating_elements: typing.List[np.ndarray],
    dates: typing.List[datetime.datetime],
    mass: float
) -> typing.List[np.ndarray]:
        states = []
        for osculating_element, d in zip(osculating_elements, dates):
            kep = orbital_state.KeplerianElements(
                sma=osculating_element[0],
                ecc=osculating_element[1],
                inc=osculating_element[2],
                arg_perigee=osculating_element[3],
                raan=osculating_element[4],
                true_anomaly=osculating_element[5],
                epoch=d,
                mu=constants.EIGEN5C_EARTH_MU
            )

            state = convert_keplerian_elements_to_orbital_state(kep, mass)
            states.append(state.get_coordinates())

        return np.array(states)

def states_to_data_frame_with_orbital_elements(
        states: typing.List[orbital_state.OrbitalState]) -> pd.DataFrame:
    e0 = states[0].get_epoch()

    dt = [s.get_epoch().get_duration_from(e0) for s in states]

    coordinates = np.array([s.get_coordinates() for s in states])

    kep = [s.get_keplerian_elements() for s in states]

    data = {
        "time": dt,
        "x": coordinates[:, 0],
        "y": coordinates[:, 1],
        "z": coordinates[:, 2],
        "xdot": coordinates[:, 3],
        "ydot": coordinates[:, 4],
        "zdot": coordinates[:, 5],
        "sma": [k.sma for k in kep],
        "ecc": [k.ecc for k in kep],
        "inc": [k.inc for k in kep],
        "arg_perigee": [k.arg_perigee for k in kep],
        "raan": [k.raan for k in kep],
        "true_anomaly": [k.true_anomaly for k in kep]
    }

    return pd.DataFrame(data)

def training_data_from_kep_with_orbital_elements(
    kep: orbital_state.KeplerianElements,
    m: float,
    step: float,
    duration: float
) -> typing.Dict:
    states = kepler_orbit_from_keplerian_elements(kep, m, step, duration)

    df = states_to_data_frame_with_orbital_elements(states)
    cartesian_orbit = df[["x", "y", "z", "xdot", "ydot", "zdot"]].to_numpy()
    kep = df[
        ["sma", "ecc", "inc", "arg_perigee", "raan", "true_anomaly"]
    ].to_numpy()
    t = df["time"].to_numpy()

    return {"t": t, "cartesian": cartesian_orbit, "kep": kep}

