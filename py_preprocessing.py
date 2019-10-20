class class_module_preprocessing:
        
    def printmd(string, color='purple'):
        """ Prints the input string in purple color. """
        from IPython.display import Markdown, display
        colorstr = "<center><b><span style='color:{}'>{}</span></center>".format(color, string)
        display(Markdown(colorstr))
    
    
    def fn_Plot_Bars(v_df, v_num, v_col_name, v_fig_size_len, v_fig_size_br):
        """ Plots the bar plot for the given input column. Takes the dataframe name, number of plot, column name, length and breadth of the figure size as the input. """     
        import matplotlib.pyplot as plt
        class_module_preprocessing.printmd("PLOT " + str(v_num) + ": " + str(v_col_name))
        print("Below is the categorical distribution: ")
        print(v_df[v_col_name].value_counts()/v_df.shape[0], "\n")
        print("\nBelow is the Bar Plot for " + str(v_col_name) + ":")
        plt.figure(figsize=(v_fig_size_len, v_fig_size_br))
        v_df[v_col_name].value_counts().plot(kind = 'bar')
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel(v_col_name, fontsize=12)
   

    def fn_Plot_Corrplot(v_corr):
        """ Plots the co-relation plot. """
        import seaborn as sns
        class_module_preprocessing.printmd("PLOT 1: Corr Plot")
        ax = sns.heatmap(
            v_corr.corr(), 
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True)
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right');
    
    
    def fn_Plot_HistSkewness(v_df, v_col_list):
        """ Plots the histogram for the numerical columns and prints the skewness. """        
        from scipy.stats import skew 
        import matplotlib.pyplot as plt
        import numpy as np
        import warnings
        warnings.filterwarnings('ignore')
        count = 12
        for v_col in v_col_list:
            print("\n")
            class_module_preprocessing.printmd("PLOT " + str(count) + ": Frequency Distribution of " + str(v_col))
            cm = plt.cm.get_cmap('cool') # Refer https://scipy-cookbook.readthedocs.io/items/Matplotlib_Show_colormaps.html
            n, bins, patches = plt.hist(v_df[v_col], 25, normed=1, color='black')
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            col = bin_centers - min(bin_centers) # Scale values to interval [0,1]
            col /= max(col)
            text_x_pos = plt.xlim()[1]-(np.abs(plt.xlim()[1]-plt.xlim()[0])*0.25)    
            plt.text(text_x_pos, plt.ylim()[1]*0.8, "Skewness: {0:0.2f}".format(skew((v_df[v_col]))))
            plt.ylabel('Number of Occurrences', fontsize=12)
            plt.xlabel(v_col, fontsize=12)
            for c, p in zip(col, patches):
                plt.setp(p, 'facecolor', cm(c))
            plt.show()
            count = count + 1

            
    def fn_ListUniqueLevelsInCatCols(v_df, v_col_list):
        """ Lists the unique level in a categorical column """
        for col in v_col_list:
            print("Number of unique levels in " + col + ": " + str(v_df[col].nunique()))
    
    
    def fn_MathTransform(df, feature):
        """ This function tests log1p, sqrt and boxcox transformations and compares their differences."""
        from scipy.stats import skew, boxcox
        import numpy as np
        print('Feature: ', feature)
        print("Skewness of log1p: ", skew(np.log1p(df[feature])))
        print("Skewness of sqrt: ", skew(np.sqrt(df[feature])))
        print("Skewness of boxcox: ", skew(boxcox(df[feature]+1)[0]))
        print("\n")

    
    def fn_MissingValuesTable(df):
        """ Prints the missing value and their percentage. """
        import pandas as pd
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("There are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.")
        return mis_val_table_ren_columns

    
    def fn_ImputeTrainData_Mode(v_train_data, v_null_cols):
        """ Imputes the train data using the mode of the column. """
        for col in v_null_cols:
            col_mode =  v_train_data[col].mode()[0]
            v_train_data[col].fillna(col_mode,inplace=True) 
        return v_train_data
    
     
    def fn_ImputeTestData_Mode(v_train_data, v_test_data, v_null_cols):
        """ Imputes the test data using the mode of train data. """
        for col in v_null_cols:
            col_mode =  v_train_data[col].mode()[0]
            v_test_data[col].fillna(col_mode,inplace=True) 
        return v_test_data
    
    
    def fn_DropCols(v_df_data, v_col_list):
        """ Drop the unimportant features from the train/test data. """
        print("\nDropping the columns: " + str(v_col_list))
        for col_name in v_col_list:
            v_df_data = v_df_data.drop([col_name], axis = 1)
        return v_df_data
    
    def fn_NoOfDaysAdmitted(v_df):
        """ Calculate a new column based on the difference between the Discharge_date and Admission_date. """
        v_df['NoOfDaysAdmitted'] = (v_df.Discharge_date - v_df.Admission_date).dt.days
        return v_df

    
    def fn_BinAge(v_df):
        """ Bin the age as a mid-point of the range. """
        age_dict = {'[0-10)':5, '[10-20)':15, '[20-30)':25, '[30-40)':35, '[40-50)':45, '[50-60)':55, '[60-70)':65, '[70-80)':75, '[80-90)':85, '[90-100)':95}
        v_df['age'] = v_df.age.map(age_dict)
        return v_df
    
    
    def fn_DrugPrescribed(v_df):
        """ Generate a new column to calculate the count of the drugs prescribed. """
        v_df['drug_prescribed'] = (v_df[['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol','troglitazone','tolazamide','insulin','glyburide.metformin','glipizide.metformin','metformin.rosiglitazone', 'metformin.pioglitazone']] != 'No').sum(1)
        return v_df
    
    
    def fn_BinDiagnosis_1(v_df):
        """ Bin the diagnosis column based on the starting character. """
        v_df.loc[v_df.diagnosis_1.str.startswith('V') | v_df.diagnosis_1.str.startswith('E'), 'diagnosis_1'] = 'OTHERS'
        v_df.loc[(v_df.diagnosis_1.str.startswith('1')), 'diagnosis_1'] = 'D1_Level1'
        v_df.loc[(v_df.diagnosis_1.str.startswith('2')), 'diagnosis_1'] = 'D1_Level2'
        v_df.loc[(v_df.diagnosis_1.str.startswith('3')), 'diagnosis_1'] = 'D1_Level3'
        v_df.loc[(v_df.diagnosis_1.str.startswith('4')), 'diagnosis_1'] = 'D1_Level4'
        v_df.loc[(v_df.diagnosis_1.str.startswith('5')), 'diagnosis_1'] = 'D1_Level5'
        v_df.loc[(v_df.diagnosis_1.str.startswith('6')), 'diagnosis_1'] = 'D1_Level6'
        v_df.loc[(v_df.diagnosis_1.str.startswith('7')), 'diagnosis_1'] = 'D1_Level7'
        v_df.loc[(v_df.diagnosis_1.str.startswith('8')), 'diagnosis_1'] = 'D1_Level8'
        v_df.loc[(v_df.diagnosis_1.str.startswith('9')), 'diagnosis_1'] = 'D1_Level9'
        return v_df

    
    def fn_BinDiagnosis_2(v_df):
        """ Bin the diagnosis column based on the starting character. """
        v_df.loc[v_df.diagnosis_2.str.startswith('V') | v_df.diagnosis_2.str.startswith('E'), 'diagnosis_2'] = 'OTHERS'
        v_df.loc[(v_df.diagnosis_2.str.startswith('1')), 'diagnosis_2'] = 'D2_Level1'
        v_df.loc[(v_df.diagnosis_2.str.startswith('2')), 'diagnosis_2'] = 'D2_Level2'
        v_df.loc[(v_df.diagnosis_2.str.startswith('3')), 'diagnosis_2'] = 'D2_Level3'
        v_df.loc[(v_df.diagnosis_2.str.startswith('4')), 'diagnosis_2'] = 'D2_Level4'
        v_df.loc[(v_df.diagnosis_2.str.startswith('5')), 'diagnosis_2'] = 'D2_Level5'
        v_df.loc[(v_df.diagnosis_2.str.startswith('6')), 'diagnosis_2'] = 'D2_Level6'
        v_df.loc[(v_df.diagnosis_2.str.startswith('7')), 'diagnosis_2'] = 'D2_Level7'
        v_df.loc[(v_df.diagnosis_2.str.startswith('8')), 'diagnosis_2'] = 'D2_Level8'
        v_df.loc[(v_df.diagnosis_2.str.startswith('9')), 'diagnosis_2'] = 'D2_Level9'
        return v_df
    
    
    def fn_BinDiagnosis_3(v_df):
        """ Bin the diagnosis column based on the starting character. """
        v_df.loc[v_df.diagnosis_3.str.startswith('V') | v_df.diagnosis_3.str.startswith('E'), 'diagnosis_3'] = 'OTHERS'
        v_df.loc[(v_df.diagnosis_3.str.startswith('1')), 'diagnosis_3'] = 'D3_Level1'
        v_df.loc[(v_df.diagnosis_3.str.startswith('2')), 'diagnosis_3'] = 'D3_Level2'
        v_df.loc[(v_df.diagnosis_3.str.startswith('3')), 'diagnosis_3'] = 'D3_Level3'
        v_df.loc[(v_df.diagnosis_3.str.startswith('4')), 'diagnosis_3'] = 'D3_Level4'
        v_df.loc[(v_df.diagnosis_3.str.startswith('5')), 'diagnosis_3'] = 'D3_Level5'
        v_df.loc[(v_df.diagnosis_3.str.startswith('6')), 'diagnosis_3'] = 'D3_Level6'
        v_df.loc[(v_df.diagnosis_3.str.startswith('7')), 'diagnosis_3'] = 'D3_Level7'
        v_df.loc[(v_df.diagnosis_3.str.startswith('8')), 'diagnosis_3'] = 'D3_Level8'
        v_df.loc[(v_df.diagnosis_3.str.startswith('9')), 'diagnosis_3'] = 'D3_Level9'
        return v_df
            
        
    def fn_BinDiagnosis(v_df, v_col, v_count):
        """ Bin the diagnosis columns. """
        v_df.loc[v_df[v_col].str.startswith('V') | v_df[v_col].str.startswith('E'), v_col] = '-99'
        v_df.loc[(v_df[v_col].str[0:3].astype('int') >= 1) & (v_df[v_col].str[0:3].astype('int') <= 139), v_col] = '-1'
        v_df.loc[(v_df[v_col].str[0:3].astype('int') >= 140) & (v_df[v_col].str[0:3].astype('int') <= 239), v_col] = '-2'
        v_df.loc[(v_df[v_col].str[0:3].astype('int') >= 240) & (v_df[v_col].str[0:3].astype('int') <= 279), v_col] = '-3'
        v_df.loc[(v_df[v_col].str[0:3].astype('int') >= 280) & (v_df[v_col].str[0:3].astype('int') <= 289), v_col] = '-4'
        v_df.loc[(v_df[v_col].str[0:3].astype('int') >= 290) & (v_df[v_col].str[0:3].astype('int') <= 319), v_col] = '-5'
        v_df.loc[(v_df[v_col].str[0:3].astype('int') >= 320) & (v_df[v_col].str[0:3].astype('int') <= 389), v_col] = '-6'
        v_df.loc[(v_df[v_col].str[0:3].astype('int') >= 390) & (v_df[v_col].str[0:3].astype('int') <= 459), v_col] = '-7'
        v_df.loc[(v_df[v_col].str[0:3].astype('int') >= 460) & (v_df[v_col].str[0:3].astype('int') <= 519), v_col] = '-8'
        v_df.loc[(v_df[v_col].str[0:3].astype('int') >= 520) & (v_df[v_col].str[0:3].astype('int') <= 579), v_col] = '-9'
        v_df.loc[(v_df[v_col].str[0:3].astype('int') >= 580) & (v_df[v_col].str[0:3].astype('int') <= 629), v_col] = '-10'
        v_df.loc[(v_df[v_col].str[0:3].astype('int') >= 630) & (v_df[v_col].str[0:3].astype('int') <= 679), v_col] = '-11'
        v_df.loc[(v_df[v_col].str[0:3].astype('int') >= 680) & (v_df[v_col].str[0:3].astype('int') <= 709), v_col] = '-12'
        v_df.loc[(v_df[v_col].str[0:3].astype('int') >= 710) & (v_df[v_col].str[0:3].astype('int') <= 739), v_col] = '-13'
        v_df.loc[(v_df[v_col].str[0:3].astype('int') >= 740) & (v_df[v_col].str[0:3].astype('int') <= 759), v_col] = '-14'
        v_df.loc[(v_df[v_col].str[0:3].astype('int') >= 760) & (v_df[v_col].str[0:3].astype('int') <= 779), v_col] = '-15'
        v_df.loc[(v_df[v_col].str[0:3].astype('int') >= 780) & (v_df[v_col].str[0:3].astype('int') <= 799), v_col] = '-16'
        v_df.loc[(v_df[v_col].str[0:3].astype('int') >= 800) & (v_df[v_col].str[0:3].astype('int') <= 999), v_col] = '-17'
        return v_df


    def fn_BinAdmissionTypeID(v_df):
        """ Bin the Admission Type ID column. """
        admission_type_id_dict = {1:1, 2:2, 
                              3:3, 4:3, 7:3, 
                              5:4, 6:4, 8:4}
        v_df.admission_type_id = v_df.admission_type_id.map(admission_type_id_dict)
        return v_df
    
    
    def fn_BinAdmissionSourceID(v_df):
        """ Bin the Admission Source ID column. """
        admission_source_id_dict = {1:1, 2:1, 3:1,
                                4:2, 5:2, 6:2, 10:2, 18:2, 19:2, 22:2, 25:2, 26:2,
                                11:3, 12:3, 13:3, 14:3, 23:3, 24:3, 7:3, 8:3,
                                9:4, 15:4, 17:4, 20:4, 21:4}

        v_df.admission_source_id = v_df.admission_source_id.map(admission_source_id_dict)
        return v_df
    
    
    def fn_BinDischargeDispositionID(v_df):
        """ Bin the Discharge Disposition ID column. """
        discharge_disposition_id_dict = {1:1, 6:1, 8:1,
                              2:2, 3:2, 4:2, 5:2, 15:2, 16:2, 17:2, 22:2, 23:2, 24:2, 30:2, 27:2, 28:2, 29:2, 
                              7:3, 9:3, 10:3, 12:3, 13:3, 14:3, 
                              11:4, 19:4, 20:4, 21:4, 
                              18:5, 25:5, 26:5}
        v_df.discharge_disposition_id = v_df.discharge_disposition_id.map(discharge_disposition_id_dict)
        return v_df
    
    
    def fn_ConvertToCatCols(v_df_data, v_col_list):
        """ Convert any column to categorical column. """
        from sklearn import preprocessing
        from sklearn.preprocessing import LabelEncoder
        for col_name in v_col_list:
            v_df_data[col_name] = v_df_data[col_name].astype('category')
        return v_df_data
    
    
    def fn_Dummify(v_df_data, v_cat_attr):
        """ Dummify categorical columns. """
        import pandas as pd
        v_df_data = pd.get_dummies(columns=v_cat_attr, data=v_df_data, drop_first=True) 
        return v_df_data
    
    
    def fn_CreateNumCols(v_df):
        """ Create 2-D Interaction columns. """
        v_df['num_LP_P'] = v_df['num_lab_procedures'] +  v_df['num_procedures'] 
        v_df['num_LP_M'] = v_df['num_lab_procedures'] +  v_df['num_medications'] 
        v_df['num_LP_D'] = v_df['num_lab_procedures'] +  v_df['num_diagnoses'] 
        v_df['num_P_M'] = v_df['num_procedures'] +  v_df['num_medications'] 
        v_df['num_P_D'] = v_df['num_procedures'] +  v_df['num_diagnoses'] 
        v_df['num_D_M'] = v_df['num_diagnoses'] +  v_df['num_medications']      
        return v_df
    

    def fn_BinDrugCols(v_df, v_col_list):
        """ Bin the drug columns. """
        for col in v_col_list:
            v_df.loc[(v_df[col] != 'No'), col] = 'OTHERS'
        return v_df


    def fn_PredictFinalData(v_clf, v_file_name, v_test_df, v_test_patiendID):
        """ Final prediction on test data. """
        import pandas as pd 
        y_pred_final = v_clf.predict(v_test_df)
        df_test_pred = pd.DataFrame(v_test_patiendID, columns = ['patientID']) 
        df_test_pred['readmitted'] = y_pred_final
        df_test_pred[['patientID', 'readmitted']].to_csv(v_file_name, index=False)
        print("Printing distribution of readmitted: \n", df_test_pred['readmitted'].value_counts())
        
        
    def fn_ReturnNumCols(v_train_data):
        """ Return the numerical columns. """
        num_cols = v_train_data.select_dtypes(include=['float64', 'int64']).columns
        return num_cols
    
    
    def fn_dt_plot_tree(tree,dataframe,label_col,plot_title):
        """ Plot the graphviz for the Decision Tree model. """
        import pandas as pd
        from sklearn.tree import export_graphviz
        label_names=pd.unique(dataframe[label_col]).astype('str')
        graph_data=export_graphviz(tree,feature_names=dataframe.drop(label_col,axis=1).columns, class_names=label_names,filled=True, rounded=True,out_file=None) # Obtaining plot data
        graph=graphviz.Source(graph_data) # Generating plot
        graph.render(plot_title)
        return graph
    
    
    def fn_PrintStats(y_train, y_pred_train, y_test, y_pred):
        """ Print the metrics. """
        from sklearn import metrics, preprocessing
        from sklearn.metrics import classification_report
        print("\nMetrics on Train Data:::::::")
        cnf_matrix = metrics.confusion_matrix(y_train, y_pred_train)
        # List of labels to index the matrix. This may be used to reorder or select a subset of labels. If none is given, those that appear at least once in y_true or y_pred are used in sorted order.
        print("Confusion Matrix:\n",cnf_matrix)
        print("Accuracy:",metrics.accuracy_score(y_train, y_pred_train))
        print("Precision:",metrics.precision_score(y_train, y_pred_train))
        print("Recall:",metrics.recall_score(y_train, y_pred_train))
        print("Classification Report:\n",classification_report(y_train, y_pred_train,digits=4))
        print("\n\nMetrics on Validation Data:::::::")
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n",cnf_matrix)
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print("Precision:",metrics.precision_score(y_test, y_pred))
        print("Recall:",metrics.recall_score(y_test, y_pred))
        print("Classification Report:\n",classification_report(y_test,y_pred,digits=4))