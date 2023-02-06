import numpy as np
import matplotlib.pyplot as plt
import pickle
import modules.data_processing as data_processing
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import pandas as pd
import scipy.stats 

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * round(y,2))

    # The percent symbol needs escaping in latex
    if mpl.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

def loss_plot(path_to_loss_data,output_path, config):
    loss_data = pd.read_csv(path_to_loss_data)
    str_list = ['Epochs:', 'Model Name:', 'Reg. Param:', 'lr:', 'BS:']

    val_loss = loss_data["Val Loss"]
    train_loss = loss_data["Train Loss"]
    conf_list = [len(train_loss),config['model_name'],config['reg_param'],config['lr'],config['batch_size']]

    plt.figure(figsize=(10,7))
    plt.title('Loss plot')
    plt.plot(train_loss[10:],color='orange',label="Train Loss")
    plt.plot(val_loss[10:],color='red',label="Validation Loss")
    for i in range(len(conf_list)):
        plt.plot([],[], ' ',label=str_list[i] + ' ' + str(conf_list[i]))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig(output_path + '_Loss_plot.pdf')
    #plt.show()


def plot(before_path,after_path):
    with open(before_path, 'rb') as handle:
        before = pickle.load(handle)
    with open(after_path, 'rb') as handle:
        after = pickle.load(handle)
    
    before = np.array(before)


    # Added because plotting is not supported for non-DataFrame objects yet. 
    if isinstance(before, pd.DataFrame) == False:
        names = ["pt","eta","phi","m","EmEnergy","HadEnergy","InvisEnergy","AuxilEnergy"]
        before = pd.DataFrame(before,columns=names)
        after = pd.DataFrame(after,columns=names)
    else: 
        pass

    ## Drop large values:
    before = before[before.pt < 8000]
    before = before[before.m < 800]
    
    after = after[after.pt < 8000]
    after = after[after.m < 800]

    print(len(before))
    print(len(after))

    Energy_before = data_processing.compute_E(before['m'], before['eta'], before['pt'])
    Energy_after = data_processing.compute_E(after['m'], after['eta'], after['pt'])

    #before = before.append(Energy_before)
    #after = after.append(Energy_after)

    DOF_2D = data_processing.compute_DOF(after)

    columns = data_processing.get_columns(before)
    number_of_columns = len(columns)



    with PdfPages(after_path.split("after.pickle")[0]+"comparison.pdf") as pdf:
        figure1, (ax1,ax2) = plt.subplots(1,2,figsize=(18.3*(1/2.54)*1.7, 13.875*(1/2.54)*1.32))
        for index, column in enumerate(columns):
            print(f'{index} of {number_of_columns}')


            response = (after-before)/before

            # Compute some information about the response
            response_norm = list(filter(lambda p : -10<=p<=10, response[column]))
            response_RMS = data_processing.RMS_function(response_norm=response_norm)
            

            # Do reduced chi^2 for the correlation plots

            chi_square_statistic = data_processing.get_chi_square_statistic(observed = after[column], expected = before[column])
            reduced_chi2 = round(chi_square_statistic/len(after[column]),5)
            print('Pearson chi^2 for',column,':',reduced_chi2)

#            minimum = int(min(before[column]+after[column]))
#            maximum = int(max(before[column]+after[column]))
#            diff = maximum - minimum
#            if diff == np.inf or diff == 0:#FIXME: We have to skip some variables
#                pdf.savefig()
#                ax2.clear() 
#                ax1.clear()
#                continue
#            step = diff/100
            #counts_before, bins_before = np.histogram(before[column],bins=np.arange(minimum,maximum,step))
            counts_before, bins_before = np.histogram(before[column],bins=np.arange(-200,500,1))
            ax1.hist(bins_before[:-1], bins_before, weights=counts_before, label='Before')
            #counts_after, bins_after = np.histogram(after[column],bins=np.arange(minimum,maximum,step))
            counts_after, bins_after = np.histogram(after[column],bins=np.arange(-200,500,1))
            ax1.hist(bins_after[:-1], bins_after, weights=counts_after, label='After',histtype='step')
            ax1.plot([], [],' ',label = fr'Pearson $\chi^2$:  {round(reduced_chi2,5)}')
            ax1.set_title(f"{column} Distribution")
            ax1.set_xlabel(column, ha='right', x=1.0)
            ax1.set_ylabel("Counts", ha='right', y=1.0)
            ax1.set_yscale('log')
            ax1.legend(loc="best")
#            minimum = min(response[column])
#            maximum = max(response[column])
#            diff = maximum - minimum
#            if diff == np.inf or diff == 0:
#                pdf.savefig()
#                ax2.clear()
#                ax1.clear() 
#                continue
#            step = diff/100
            #counts_response, bins_response = np.histogram(response[column],bins=np.arange(minimum,maximum,step))
            counts_response, bins_response = np.histogram(response[column],bins=np.arange(-2,2,0.1))
            ax2.hist(bins_response[:-1], bins_response, weights=counts_response, label='Response')
            ax2.axvline(np.mean(response_norm), color='k', linestyle='dashed', linewidth=1,label=f'Mean: {round(np.mean(response_norm),8)}')
            ax2.plot([], [],' ',label = f'RMS: {round(response_RMS,8)}')
            #To have percent on the x-axis
            #formatter = mpl.ticker.FuncFormatter(to_percent)
            #ax2.xaxis.set_major_formatter(formatter)   
            ax2.set_title(f"{column} Response")
            ax2.set_xlabel(f'{column} Response', ha='right', x=1.0)
            ax2.set_ylabel("Counts", ha='right', y=1.0)
            ax2.legend(loc='best')

            figure1.set_tight_layout(True)
            pdf.savefig()
            ax2.clear()
            ax1.clear()
            
            #if index==1: break

