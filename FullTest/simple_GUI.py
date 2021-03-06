import PySimpleGUI as sg
import datetime

def run_GUI():
    sg.ChangeLookAndFeel('White')
    Y=10
    date_N_years_ago = datetime.datetime.now() - datetime.timedelta(days=Y*365)
    date_N_years_ago=date_N_years_ago.strftime("%Y-%m-%d 00:00:00")
    N=1
    date_N_days_ago = datetime.datetime.now() - datetime.timedelta(days=N)
    date_N_days_ago=date_N_days_ago.strftime("%Y-%m-%d 00:00:00")
    column1 = [[sg.Text('Choose Start Date', background_color='#d3dfda', justification='center', size=(20, 1))],      
               [sg.In(date_N_years_ago, size=(20,1), key='input1')],
               [sg.CalendarButton('Choose Date', target='input1', key='date1')]]  
    column2 = [[sg.Text('Choose End Date', background_color='#d3dfda', justification='center', size=(20, 1))],      
               [sg.In(date_N_days_ago, size=(20,1), key='input2')],
               [sg.CalendarButton('Choose Date', target='input2', key='date2')]]      

    layout = [      
        [sg.Text('AIinFinance', size=(30, 1), font=("Helvetica", 25))],      
        [sg.InputCombo(('AAPL', 'GOOGL', 'HP'), size=(20, 3), key='text1')],     
        #[sg.InputText('This is my text')],
        [sg.Column(column1, background_color='#d3dfda'),sg.Column(column2, background_color='#d3dfda')],
        [sg.Ok(key=1)]
    ]
    """
        [sg.Checkbox('My first checkbox!'), sg.Checkbox('My second checkbox!', default=True)],      
        [sg.Radio('My first Radio!     ', "RADIO1", default=True), sg.Radio('My second Radio!', "RADIO1")],      
        [sg.Multiline(default_text='This is the default Text should you decide not to type anything', size=(35, 3)),      
         sg.Multiline(default_text='A second multi-line', size=(35, 3))],      
        [sg.InputCombo(('Combobox 1', 'Combobox 2'), size=(20, 3)),      
         sg.Slider(range=(1, 100), orientation='h', size=(34, 20), default_value=85)],      
        [sg.Listbox(values=('Listbox 1', 'Listbox 2', 'Listbox 3'), size=(30, 3)),      
         sg.Slider(range=(1, 100), orientation='v', size=(5, 20), default_value=25),      
         sg.Slider(range=(1, 100), orientation='v', size=(5, 20), default_value=75),      
         sg.Slider(range=(1, 100), orientation='v', size=(5, 20), default_value=10),      
         sg.Column(column1, background_color='#d3dfda')],      
        [sg.Text('_'  * 80)],      
        [sg.Text('Choose A Folder', size=(35, 1))],      
        [sg.Text('Your Folder', size=(15, 1), auto_size_text=False, justification='right'),      
         sg.InputText('Default Folder'), sg.FolderBrowse()],      
        [sg.Submit(), sg.Cancel()]
        """
    window = sg.Window('Everything bagel', default_element_size=(40, 1)).Layout(layout)
    button, values = window.Read()

    f= open("input.txt","w+")
    f.write(values["text1"]+" ")
    f.write(values["input1"]+" ")
    f.write(values["input2"]+" ")
    f.close()
    #sg.Popup(button, values)

def main():
    run_GUI()

main()
