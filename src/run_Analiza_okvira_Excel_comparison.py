import Analiza_okvira_v0_33 as ao
import numpy as np


if __name__ == '__main__': #__name__='__main__' samo ako smo pokrenuli ovaj file! Ako smo ga importirali, onda nije! 



    
    InputFile=ao.Input_file('ponton_starting_excell_configuration.txt',480)
    InputFile.load_file()
    model = InputFile.create_model()            # u modelu je spremljen structure_obj preko kojeg imamo pristup svim funkcijama i kreiranim objektima
    
    ao.calculate_problem(model)

    for beam in model.beams:
        print(f'\nMoment u cvoru {beam.node1.ID}-{beam.node2.ID}: {beam.M12}')
        print(f'Moment u cvoru {beam.node2.ID}-{beam.node1.ID}: {beam.M21}')
        print(f'Moment otpora: {beam.prop.sect.Wy}')
        print(f'Naprezanje: {beam.max_s}\n')

    print(f'Masa konstrukcije: {model.get_mass()}')

trapezius = model.beams[0].intrinsic_diagram_w_trap - model.beams[0].intrinsic_diagram

print(trapezius)

trapezius = model.beams[2].intrinsic_diagram_w_trap - model.beams[2].intrinsic_diagram

print(trapezius)


trapezius = model.beams[5].intrinsic_diagram_w_trap - model.beams[5].intrinsic_diagram

print(trapezius)

for beam in model.beams:
    print(f'beam.M12: {beam.M12} = beam.intrinsic_diagram_w_trap[0]: {beam.intrinsic_diagram_w_trap[0]}')


    
    
