import numpy as np

filepath = 'dehumanization-dictionary.dicx'
new_filepath = 'dehumanization-dictionary.dic'

new_content = '%\n'

first_row = True

separator = '\t'

with open(filepath) as lines:
    
    for line in lines:
        if first_row: # the first line is about the categories
            category_mapping = line.split(',')[1:]
            
            for i, category in enumerate(category_mapping):
                category = category.replace('\n', '')
                new_content += str(i+1)+separator+category+'\n'
                
            new_content += '%\n'

            first_row = False

        else:    
            
            content = line.replace('\n', '').split(',')

            word = content.pop(0)

            np_array = np.array(content)
            item_index = [str(index+1) for index in np.where(np_array=='X')[0].tolist()]

            new_content += word+separator+separator.join(item_index)+'\n'
        
f = open(new_filepath, 'w')
f.write(new_content)
f.close()
