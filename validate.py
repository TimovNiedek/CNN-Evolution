# Gegeven een variabele, returnt True als het van een goede vorm is volgens ons concept
# Anders False
def checkNetwork(network):
    # Netwerk moet een dictionary zijn
    if not type(network) is dict:
        print('ERROR: Netwerk is geen dictionary');
        return False;
    # Netwerkdictionary moet block32 key bevatten
    if not 'block32' in network:
        print('ERROR: Netwerk bevat geen block32');
        return False;
    else:
        # Als er een block32 definitie is, moet deze voldoen aan de block standaard
        if not checkBlock(network['block32'], '32'):
            print('ERROR: Block32 voldoet niet aan de structuur');
            return False;
    
    # Netwerkdictionary moet subsample32 key bevatten
    if not 'subsample32' in network:
        print('ERROR: Netwerk bevat geen subsample32');
        return False;
    else:
        # Het type van de waarde van de subsample32 key moet boolean zijn
        if not type(network['subsample32']) is bool:
            print('ERROR: Subsample32 is geen boolean');
            return False;
    
    # Netwerkdictionary moet block16 key bevatten
    if not 'block16' in network:
        print('ERROR: Netwerk bevat geen block16');
        return False;
    else:
        # Als er een block16 definitie is, moet deze voldoen aan de block standaard
        if not checkBlock(network['block16'], '16'):
            print('ERROR: Block16 voldoet niet aan de structuur');
            return False;
                  
    # Netwerkdictionary moet subsample16 key bevatten
    if not 'subsample16' in network:
        print('ERROR: Netwerk bevat geen subsample16');
        return False;
    else:
        # Het type van de waarde van de subsample16 key moet boolean zijn
        if not type(network['subsample16']) is bool:
            print('ERROR: Subsample16 is geen boolean');
            return False;
    
    # Netwerkdictionary moet block8 key bevatten
    if not 'block8' in network:
        print('ERROR: Netwerk bevat geen block8');
        return False;
    else:
        # Als er een block8 definitie is, moet deze voldoen aan de block standaard
        if not checkBlock(network['block8'], '8'):
            print('ERROR: Block8 voldoet niet aan de structuur');
            return False;
                  
    # Netwerkdictionary moet subsample8 key bevatten
    if not 'subsample8' in network:
        print('ERROR: Netwerk bevat geen subsample8');
        return False;
    else:
        # Het type van de waarde van de subsample8 key moet boolean zijn
        if not type(network['subsample8']) is bool:
            print('ERROR: Subsample8 is geen boolean');
            return False;
                  
    # Netwerkdictionary moet final key bevatten
    if not 'final' in network:
        print('ERROR: Netwerk bevat geen final');
        return False;
    else:
        # Het type van de waarde van de final key moet boolean zijn
        if not type(network['final']) is bool:
            print('ERROR: Final is geen boolean');
            return False;
    
    return True;

# Gegeven een variabele, geeft aan of het voldoet aan de blockstructuur voor een netwerk
# Returnt True als dit zo is, anders False
def checkBlock(block, string):
    # Block moet een dictionary zijn
    if not type(block) is dict:
        print('ERROR: Block ' + string + ' is geen dictionary');
        return False;
    
    # Block moet definitie voor convlayers bevatten
    if not 'convblocks' in block:
        print('ERROR: Block ' + string + ' bevat geen convblocks');
        return False;
    else:
        for layer in block['convblocks']:
            # Elke layer in conblocks moet voldoen aan de goede structuur
            if not checkLayer(layer):
                print('ERROR: Block ' + string + ' bevat een layer die niet aan de structuur voldoet');
                return False;
            # Block moet definitie voor output bevatten
            if not 'output' in block:
                print('ERROR: Block ' + string + ' bevat geen output');
                return False;
            else:
                # als de layer niet in de output voorkomt
                if not layer['id'] in block['output']:
                    used = False;
                    # voor alle andere layers
                    for layer2 in block['convblocks']:
                        # als het id van deze layer voorkomt in de input van de andere layer
                        # en het is niet dezelfde layer, dan is voldoet de structuur, anders niet
                        # (check of alle outputs ergens worden gebruikt)
                        if not layer['id'] == layer2['id'] and layer['id'] in layer2['input']:
                            used = True;
                    if (not used):
                        print('ERROR: Block ' + string + 'gebruikt niet elke layer');
                    return used;
                return True;
            
# Gegeven een variabele, geeft aan of het voldoet aan de structuur van een layer/convlock
# Returnt True als dit zo is, anders False
def checkLayer(layer):
    # Layer moet een dictionary zijn
    if not type(layer) is dict:
        print('ERROR: Layer is geen dictionary');
        return False;
    # Layer moet definitie voor id hebben
    if not 'id' in layer:
        print('ERROR: Layer bevat geen id');
        return False;
    else:
        # id moet een integer zijn
        if not type(layer['id']) is int:
            print('ERROR: Layer id is geen int');
            return False;
        else:
            # id moet 0, 1 of 2 zijn
            if layer['id'] < 0 or layer['id'] > 2:
                print('ERROR: Layer ' + layer['id'] + ' heeft ongeldig id');
                return False;
        
        # Layer moet definitie voor layers hebben
        if not 'layers' in layer:
            print('ERROR: Layer ' + layer['id'] + ' heeft geen layers');
            return False;
        else:
            # Elk onderdeel van de layer moet either, BN, Conv, Dropout of Relu zijn
            for part in layer['layers']:
                if (not part == 'b') and (not part == 'c') and (not part == 'd') and (not part == 'r'):
                    print('ERROR: Layer ' + layer['id'] + ' heeft ongeldige layer(s)');
                    return False;
            # Elke layer moet BN, Conv en Relu bevatten
            if not 'b' in layer['layers']:
                print('ERROR: Layer ' + layer['id'] + ' heeft geen Batch Normalization');
                return False;
            if not 'c' in layer['layers']:
                print('ERROR: Layer ' + layer['id'] + ' heeft geen Convolution');
                return False;
            if not 'r' in layer['layers']:
                print('ERROR: Layer ' + layer['id'] + ' heeft geen Relu');
                return False;
            
            # Layer moet definitie voor input hebben
            if not 'input' in layer:
                print('ERROR: Layer ' + layer['id'] + ' heeft geen input');
                return False;
            else:
                # input kan alleen -1, 0 of 1 zijn
                for layerID in layer['input']:
                    if layerID < -1 or layerID > 1:
                        print('ERROR: Layer ' + layer['id'] + ' input bevat ongeldig id');
                        return False;
                    # input mag niet hoger zijn dan de id van de layer zelf
                    if layerID > layer['id']:
                        print('ERROR: Layer ' + layer['id'] + ' input bevat hoger id');
                        return False;
    return True;
