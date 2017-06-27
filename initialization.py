import random;

# Functie om een heel nieuw random netwerk te initializeren
# secondLayer en thirdLayer is de kans om respectievelijk 2 of 3 lagen in een block te maken
# subsample en final is de kans dat respectievelijk de subsample en final booleans True is
# output is de kans dat een layer die niet in de output hoeft (dus ergens anders al gebruikt wordt) toch in de output van een block voorkomt
# dropout is de kans dat een layer/conblock een dropout layer heeft of niet
def initializeNetwork(secondLayer=50, thirdLayer=50, subsample=50, final=50, output=50, dropout=50):
    block32 = createRandomBlock(secondLayer, thirdLayer, output, dropout);
    subsample32choice = getRandomBoolean(subsample);
    block16 = createRandomBlock(secondLayer, thirdLayer, output, dropout);
    subsample16choice = getRandomBoolean(subsample);
    block8 = createRandomBlock(secondLayer, thirdLayer, output, dropout);
    subsample8choice = getRandomBoolean(subsample);
    finalchoice = getRandomBoolean(final);
    
    return {'block32': block32,
          'subsample32': subsample32choice,
          'block16': block16,
          'subsample16': subsample16choice,
          'block8': block8,
          'subsample8': subsample8choice,
          'final': finalchoice
         };

# Functie om een random block te initializeren
def createRandomBlock(secondLayer, thirdLayer, outputChance, dropoutChance):
    # Eerste layer bestaat sowieso
    layer0 = createRandomLayer(0, [[-1]], dropoutChance);
    layers = [layer0];
    # 50% kans voor een tweede layer
    if (getRandomBoolean(secondLayer)):
        layer1 = createRandomLayer(1, [[-1],[0]], dropoutChance);
        layers.append(layer1);
        # Als er een tweede layer is, nog eens 50% kans op een derde layer
        if (getRandomBoolean(thirdLayer)):
            layer2 = createRandomLayer(2, [[-1],[0],[1],[0,1]], dropoutChance);
            layers.append(layer2);
                         
    # Stop alle layers wiens output niet gebruikt wordt in een andere layer
    # sowieso in de output van het block
    output = [];
    for layer in layers:
        used = False;
        for layer2 in layers:
            if (layer['id'] in layer2['input']):
                used = True;
                break;
        if (not used):
            output.append(layer['id']);
                
    # Voor alle andere layers hebben deze 50% kans om ook in de output te komen
    for layer in layers:
        if (layer['id'] not in output):
            if (getRandomBoolean(outputChance)):
                output.append(layer['id']);
                
    return {'convblocks': layers,
        'output': output
        }
    
# Functie om een random layer te initializeren
def createRandomLayer(idn, posInputs, dropoutChance):
    # Batch normalization, convolution en relu zitten sowieso in de layer
    layers = ['b', 'c', 'r'];
             
    # 50% kans om dropout in de layer te stoppen
    if (getRandomBoolean(dropoutChance)):
        layers.append('d');
    
    # Willekeurige volgorde van de componenten van de layer
    random.shuffle(layers);
               
    # Kies willekeurige mogelijkheid voor de inputs
    inputs = random.choice(posInputs);
            
    return {'id': idn, 'layers': layers, 'input': inputs}
                
    
# Functie om willekeurge boolean te krijgen 
# Met percent kan de kans worden aangegeven op True (standaard 50)
# Hoe hoger percent hoe hoger de kans op true
def getRandomBoolean(percent=50):
    return random.randrange(100) < percent;