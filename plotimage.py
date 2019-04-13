import numpy as np
import matplotlib.pyplot as plt
import load_mnist

train_set, train_results, test_set, test_results = load_mnist.load_data()

indices = []
index0 = [j  for j in range(10)]
index1 = [j+6000  for j in range(10)]
index2 = [j+6000*3  for j in range(10)]
index3 = [j+6000*4  for j in range(10)]
index4 = [j+6000*5  for j in range(10)]
index5 = [j+6000*6  for j in range(10)]
index6 = [j+38000  for j in range(10)]
index7 = [j+6000*7  for j in range(10)]
index8 = [j+6000*9  for j in range(10)]
index9 = [j+58000  for j in range(10)]

indices = index0+index1+index2+index3+index4+index5+index6+index7+index8+index9





fig, axes = plt.subplots(10, 10, figsize = (20, 20), sharex = True, sharey = True)
plt.rc('axes', titlesize=12)

for i in range(100):
    subplot_row = i//10
    subplot_col = i%10
    ax = axes[subplot_row, subplot_col]
    
    plottable_image = np.reshape(train_set[indices[i]]*255., (28, 28))
    ax.imshow(plottable_image, cmap='gray_r')
    
    ax.set_title('Digit Label: {}'.format(np.argmax(train_results[indices[i]])))
    ax.set_xbound([0, 28])
 
plt.tight_layout(True)
#plt.show()
plt.savefig('part1image.pdf')                 
                 
                 
                 
