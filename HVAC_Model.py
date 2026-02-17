import pyomo.environ as pyo
import numpy as np

# Import data from LoadingData.py
from LoadingData import occupancy_room1, occupancy_room2, price_data
from SystemCharacteristics import get_fixed_data
#Whatever
def create_HVAC_model(scenario_idx=0):
    """
    Create Pyomo model for HVAC optimization.
    scenario_idx: index to select which scenario (row) from the data files
    """
    
    # Get fixed system parameters
    params = get_fixed_data()
    num_timeslots = params['num_timeslots']
    
    # Create model
    model = pyo.ConcreteModel("HVAC_Optimization")
    
    # ==============================
    # Sets
    # ==============================
    model.T = pyo.RangeSet(0, num_timeslots - 1)  # Time periods
    model.R = pyo.RangeSet(1, 2)  # Rooms (1 and 2)
    
    # ==============================
    # Parameters (from data files)
    # ==============================
    
    # Occupancy level in room r at time t [number of people]
    model.O = pyo.Param(model.R, model.T, initialize=lambda m, r, t: 
        occupancy_room1[scenario_idx, t] if r == 1 else occupancy_room2[scenario_idx, t])
    
    # Electricity price at time t [€/kWh]
    model.lambda_t = pyo.Param(model.T, initialize=lambda m, t: price_data[scenario_idx, t])
    
    # Outdoor temperature at time t [°C]
    model.temp_out = pyo.Param(model.T, initialize=lambda m, t: params['outdoor_temperature'][t])
    
    # System coefficients
    model.zeta_exch = pyo.Param(initialize=params['heat_exchange_coeff'])  # Heat exchange between rooms
    model.zeta_loss = pyo.Param(initialize=params['thermal_loss_coeff'])   # Thermal loss coefficient
    model.zeta_heat = pyo.Param(initialize=params['heating_efficiency_coeff'])  # Heater power effect
    model.zeta_occ = pyo.Param(initialize=params['heat_occupancy_coeff'])  # Occupancy heat gain
    model.zeta_vent = pyo.Param(initialize=params['heat_vent_coeff'])  # Ventilation cooling effect
    
    model.eta_occ = pyo.Param(initialize=params['humidity_occupancy_coeff'])  # Humidity increase per person
    model.eta_vent = pyo.Param(initialize=params['humidity_vent_coeff'])  # Humidity reduction from ventilation
    
    model.P_vent = pyo.Param(initialize=params['ventilation_power'])  # Ventilation power consumption
    model.P_max = pyo.Param(initialize=params['heating_max_power'])  # Maximum heater power
    
    model.T_Low = pyo.Param(initialize=params['temp_min_comfort_threshold'])  # Lower comfort threshold
    model.T_OK = pyo.Param(initialize=params['temp_OK_threshold'])  # OK threshold (overrule deactivation)
    model.T_High = pyo.Param(initialize=params['temp_max_comfort_threshold'])  # Upper comfort threshold
    model.H_high = pyo.Param(initialize=params['humidity_threshold'])  # Humidity threshold
    
    model.temp_init = pyo.Param(initialize=params['initial_temperature'])
    model.hum_init = pyo.Param(initialize=params['initial_humidity'])
    
    # Big-M for constraints
    model.M = pyo.Param(initialize=100)
    
    # ==============================
    # Variables
    # ==============================
    
    # Temperature of room r at time t [°C]
    model.temp = pyo.Var(model.R, model.T, domain=pyo.Reals, bounds=(0, 50))
    
    # Humidity level of the whole place at time t [%]
    model.hum = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, 100))
    
    # Power consumption of heater in room r at time t [kW]
    model.p = pyo.Var(model.R, model.T, domain=pyo.NonNegativeReals, bounds=(0, params['heating_max_power']))
    
    # Ventilation status at time t (binary: 0=OFF, 1=ON)
    model.v = pyo.Var(model.T, domain=pyo.Binary)
    
    # Binary: ventilation switched OFF at time t (0=not switched off, 1=switched off)
    model.V_off = pyo.Var(model.T, domain=pyo.Binary)
    
    # Binary: ventilation switched ON at time t (0=not switched on, 1=switched on)
    model.V_on = pyo.Var(model.T, domain=pyo.Binary)
    
    # Binary for Override Mode at lower temperatures (1=ON, 0=OFF)
    model.OM = pyo.Var(model.R, model.T, domain=pyo.Binary)
    
    # Binary for switch off of Override Mode
    model.OM_off = pyo.Var(model.R, model.T, domain=pyo.Binary)
    
    # Binary to know whether we are above T^OK (0=Below, 1=Above)
    model.delta_OK = pyo.Var(model.R, model.T, domain=pyo.Binary)
    
    # Binary to know whether we are above T^High (0=Below, 1=Above)
    model.eta_High = pyo.Var(model.R, model.T, domain=pyo.Binary)
    
    # ==============================
    # Objective Function (eq 3.2)
    # Minimize total electricity cost
    # ==============================
    def objective_rule(m):
        return sum(m.lambda_t[t] * (m.p[1, t] + m.p[2, t] + m.P_vent * m.v[t]) for t in m.T)
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    
    # ==============================
    # Constraints
    # ==============================
    
    # --- 3.3.1 Temperature Equation (eq 1) ---
    def temperature_rule(m, r, t):
        if t == 0:
            # First hour: temperature equals initial value
            return m.temp[r, t] == m.temp_init
        else:
            temp_prev = m.temp[r, t-1]
            other_room = 2 if r == 1 else 1
            other_temp_prev = m.temp[other_room, t-1]
            v_prev = m.v[t-1]
            O_prev = m.O[r, t-1]
        
            return m.temp[r, t] == (
                temp_prev
                + m.zeta_exch * (other_temp_prev - temp_prev)  # heat exchange between rooms
                + m.zeta_loss * (m.temp_out[t-1] - temp_prev)  # heat loss to outside
                + m.zeta_heat * m.p[r, t-1]  # heater effect
                - m.zeta_vent * v_prev  # ventilation cooling
                + m.zeta_occ * O_prev  # occupancy heat gain
            )
    model.temperature_constraint = pyo.Constraint(model.R, model.T, rule=temperature_rule)
    
    # --- 3.3.2 Temperature Overrule Controller ---
    
    # Heater bounds (eq 3): 0 <= p_{r,t} <= P_max (already in variable bounds)
    
    # Lower Temperature Bound - Overrule ensures power at max when OM is on (eq 4)
    def overrule_power_rule(m, r, t):
        return m.p[r, t] >= m.OM[r, t] * m.P_max
    model.overrule_power = pyo.Constraint(model.R, model.T, rule=overrule_power_rule)
    
    # Trigger for overrule mode when temp below T_Low (eq 5)
    def overrule_trigger_rule(m, r, t):
        return m.T_Low - m.temp[r, t] <= m.M * m.OM[r, t]
    model.overrule_trigger = pyo.Constraint(model.R, model.T, rule=overrule_trigger_rule)
    
    # Exit path - detect when temp reaches T_OK (eq 6)
    def temp_ok_indicator_rule(m, r, t):
        return m.T_OK - m.temp[r, t] <= m.M * (1 - m.delta_OK[r, t])
    model.temp_ok_indicator = pyo.Constraint(model.R, model.T, rule=temp_ok_indicator_rule)
    
    # OM_off can only be 1 if we're above T_OK (eq 7)
    def om_off_condition_rule(m, r, t):
        return m.OM_off[r, t] <= m.delta_OK[r, t]
    model.om_off_condition = pyo.Constraint(model.R, model.T, rule=om_off_condition_rule)
    
    # OM state transition (eq 8)
    def om_transition_rule(m, r, t):
        if t == 0:
            # Use the known initial temperature parameter (not the variable)
            if pyo.value(m.temp_init) < pyo.value(m.T_Low):
                return m.OM[r, t] == 1  # start in overrule mode
            else:
                return m.OM[r, t] == 0  # start with overrule off
        return m.OM[r, t] == m.OM[r, t-1] - m.OM_off[r, t]
    model.om_transition = pyo.Constraint(model.R, model.T, rule=om_transition_rule)
    
    # OM_off <= OM_{t-1} (can only switch off if it was on)
    def om_off_limit_rule(m, r, t):
        if t == 0:
            return m.OM_off[r, t] == 0
        return m.OM_off[r, t] <= m.OM[r, t-1]
    model.om_off_limit = pyo.Constraint(model.R, model.T, rule=om_off_limit_rule)
    
    # Upper Temperature Bound - detect when above T_High (eq 8 upper)
    def temp_high_indicator_rule(m, r, t):
        return m.temp[r, t] - m.T_High <= m.M * m.eta_High[r, t]
    model.temp_high_indicator = pyo.Constraint(model.R, model.T, rule=temp_high_indicator_rule)
    
    # Heater must be off when above T_High (eq 9)
    def heater_off_high_temp_rule(m, r, t):
        return m.p[r, t] <= (1 - m.eta_High[r, t]) * m.P_max
    model.heater_off_high_temp = pyo.Constraint(model.R, model.T, rule=heater_off_high_temp_rule)
    
    # --- 3.3.3 Humidity Equation (eq 10) ---
    def humidity_rule(m, t):
        if t == 0:
            # First hour: humidity equals initial value
            return m.hum[t] == m.hum_init
        else:
            hum_prev = m.hum[t-1]
            v_prev = m.v[t-1]
            O_total_prev = m.O[1, t-1] + m.O[2, t-1]
        
            return m.hum[t] == hum_prev - m.eta_vent * v_prev + m.eta_occ * O_total_prev
    model.humidity_constraint = pyo.Constraint(model.T, rule=humidity_rule)
    
    # --- 3.3.4 Humidity Overrule Controller (eq 11) ---
    def humidity_overrule_rule(m, t):
        return m.hum[t] - m.H_high <= m.M * m.v[t]
    model.humidity_overrule = pyo.Constraint(model.T, rule=humidity_overrule_rule)
    
    # --- 3.3.5 Ventilation Operational Constraints ---
    
    # v_t = v_{t-1} + V_on_t - V_off_t
    def vent_state_rule(m, t):
        if t == 0:
            return m.v[t] == 0 + m.V_on[t] - m.V_off[t]  # initial v=0
        return m.v[t] == m.v[t-1] + m.V_on[t] - m.V_off[t]
    model.vent_state = pyo.Constraint(model.T, rule=vent_state_rule)
    
    # Can't switch on and off at the same time
    def vent_switch_rule(m, t):
        return m.V_on[t] + m.V_off[t] <= 1
    model.vent_switch = pyo.Constraint(model.T, rule=vent_switch_rule)
    
    # V_off_t <= v_{t-1} (can only switch off if it was on)
    def vent_off_limit_rule(m, t):
        if t == 0:
            return m.V_off[t] <= 0
        return m.V_off[t] <= m.v[t-1]
    model.vent_off_limit = pyo.Constraint(model.T, rule=vent_off_limit_rule)
    
    # V_on_t <= 1 - v_{t-1} (can only switch on if it was off)
    def vent_on_limit_rule(m, t):
        if t == 0:
            return m.V_on[t] <= 1
        return m.V_on[t] <= 1 - m.v[t-1]
    model.vent_on_limit = pyo.Constraint(model.T, rule=vent_on_limit_rule)
    
    # --- 3.3.6 Ventilation System Inertia ---
    # If ventilation is switched on, it must stay on for at least 3 hours
    def vent_inertia_rule(m, t):
        if t > num_timeslots - 3:
            return pyo.Constraint.Skip  # not enough time periods left
        return 3 * m.V_on[t] <= m.v[t] + m.v[t+1] + m.v[t+2]
    model.vent_inertia = pyo.Constraint(model.T, rule=vent_inertia_rule)
    
    return model


import pandas as pd

# Example usage
if __name__ == "__main__":
    # Solve for all 100 days and calculate average cost
    solver = pyo.SolverFactory('gurobi')  
    
    num_days = 100
    daily_costs = []
    all_results = []  # Store detailed results for each day
    
    for day in range(num_days):
        model = create_HVAC_model(scenario_idx=day)
        results = solver.solve(model, tee=False)
        
        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            cost = pyo.value(model.obj)
            daily_costs.append(cost)
            print(f"Day {day+1}: Cost = {cost:.4f} €")
            
            # Store detailed results for this day
            for t in model.T:
                all_results.append({
                    'Day': day + 1,
                    'Hour': t,
                    'Electricity_Price_EUR_kWh': pyo.value(model.lambda_t[t]),
                    'Temp_Room1_C': pyo.value(model.temp[1, t]),
                    'Temp_Room2_C': pyo.value(model.temp[2, t]),
                    'Heater_Room1_kW': pyo.value(model.p[1, t]),
                    'Heater_Room2_kW': pyo.value(model.p[2, t]),
                    'Ventilation_ON': int(pyo.value(model.v[t])),
                    'Humidity_percent': pyo.value(model.hum[t]),
                    'Occupancy_Room1': pyo.value(model.O[1, t]),
                    'Occupancy_Room2': pyo.value(model.O[2, t]),
                    'OverrideMode_Room1': int(pyo.value(model.OM[1, t])),
                    'OverrideMode_Room2': int(pyo.value(model.OM[2, t])),
                })
        else:
            print(f"Day {day+1}: Solver failed - {results.solver.termination_condition}")
    
    # Calculate average daily electricity cost
    if daily_costs:
        average_cost = np.mean(daily_costs)
        print(f"\n=== RESULTS ===")
        print(f"Number of days solved: {len(daily_costs)}")
        print(f"Average daily electricity cost: {average_cost:.4f} €")
        print(f"Min daily cost: {min(daily_costs):.4f} €")
        print(f"Max daily cost: {max(daily_costs):.4f} €")
    
    # ----- Export results to Excel -----
    excel_file = 'HVAC_Results.xlsx'
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Sheet 1: Detailed hourly results
        df_detailed = pd.DataFrame(all_results)
        df_detailed.to_excel(writer, sheet_name='Hourly_Results', index=False)
        
        # Sheet 2: Daily summary
        df_daily = pd.DataFrame({
            'Day': list(range(1, len(daily_costs) + 1)),
            'Daily_Cost_EUR': daily_costs

        })
        df_daily.to_excel(writer, sheet_name='Daily_Costs', index=False)
        
        # Sheet 3: Summary statistics
        df_summary = pd.DataFrame({
            'Metric': ['Average Daily Cost (EUR)', 'Min Daily Cost (EUR)', 
                       'Max Daily Cost (EUR)', 'Total Days Solved'],
            'Value': [average_cost, min(daily_costs), max(daily_costs), len(daily_costs)]
        })
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"\nResults saved to Excel: {excel_file}")
    print("  - Sheet 'Hourly_Results': Detailed hourly data for all days")
    print("  - Sheet 'Daily_Costs': Daily electricity costs")
    print("  - Sheet 'Summary': Summary statistics")
    
    # ----- Plot results for a selected day -----
    selected_day = 50
    print(f"\n--- Plotting results for Day {selected_day + 1} ---")
    model = create_HVAC_model(scenario_idx=selected_day)
    solver.solve(model, tee=False)
    
    # Print temperature and overrule mode for each room at each hour
    print(f"\nTemperatures and Overrule Mode for Day {selected_day + 1}:")
    print(f"{'Hour':<6} {'Room 1 (°C)':<15} {'Room 2 (°C)':<15} {'OM Room 1':<12} {'OM Room 2':<12}")
    print("-" * 60)
    for t in model.T:
        temp_r1 = pyo.value(model.temp[1, t])
        temp_r2 = pyo.value(model.temp[2, t])
        om_r1 = int(pyo.value(model.OM[1, t]))
        om_r2 = int(pyo.value(model.OM[2, t]))
        print(f"{t:<6} {temp_r1:<15.2f} {temp_r2:<15.2f} {om_r1:<12} {om_r2:<12}")
    
    # Get initial values from parameters
    params = get_fixed_data()
    temp_init = params['initial_temperature']
    hum_init = params['initial_humidity']
    
    # Extract results directly from model (t=0 already uses initial values in constraints)
    HVAC_results = {
        'Temp_r1': [pyo.value(model.temp[1, t]) for t in model.T],
        'Temp_r2': [pyo.value(model.temp[2, t]) for t in model.T],
        'h_r1': [pyo.value(model.p[1, t]) for t in model.T],
        'h_r2': [pyo.value(model.p[2, t]) for t in model.T],
        'v': [pyo.value(model.v[t]) for t in model.T],
        'Hum': [pyo.value(model.hum[t]) for t in model.T],
        'price': [pyo.value(model.lambda_t[t]) for t in model.T],
        'Occ_r1': [pyo.value(model.O[1, t]) for t in model.T],
        'Occ_r2': [pyo.value(model.O[2, t]) for t in model.T],
    }
    
    from PlotsRestaurant import plot_HVAC_results
    plot_HVAC_results(HVAC_results)
