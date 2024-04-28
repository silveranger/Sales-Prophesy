import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import prediction
import plotly.express as px
from streamlit_js_eval import streamlit_js_eval
import streamlit.components.v1 as components


st.set_page_config(
    page_title="SalesProphecy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.google.co.in/',
        'Report a bug': "https://www.google.co.in/",
        'About': "# Sales Prophecy \n This is a sales prediction app!\nIt combines powerful models to provide an accurate forecast.\nThe dataset must have 2 columns date and sales."
    }
)



st.markdown("<div style='border: 2px solid #000; border-radius: 10px; padding: 20px; border-color: #FF4B4B; background-color: #FF4B4B';'><h1 style='text-align: center; color: white;'>📈 Sales Prophecy</h1></div><br><br><br>", unsafe_allow_html=True)
def streamlit_menu():
    selected = option_menu(
        menu_title=None,  # required
        options=["Home", "Product", "Predict"],  # required
        icons=["house", "box", "activity"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
        orientation="horizontal",
        styles={
            # "container": {"padding": "0!important", "background-color": "#fafafa"},
            # "icon": {"color": "orange", "font-size": "25px"},
            "nav-link": {
                "font-size": "25px",
                "padding": "10px",
                # "text-align": "left",
                # "margin": "10px",
                # "--hover-color": "#eee",
            },
            # "nav-link-selected": {"background-color": "green"},
        },
    )
    return selected



selected = streamlit_menu()

if selected == "Home":

    def Pie(data):
        data['date'] = pd.to_datetime(data['date'], format="%d-%m-%Y")
        res = pd.DataFrame(data.groupby(data['date'].dt.strftime('%B'))['sales'].sum())
        # fig, ax = plt.subplots(figsize=(10, 8))
        # sns.boxplot(data=res, x='date', y='sales',palette='cyan')
        # ax.set_title('Sales on Month')
        st.write(f'Monthly Distribution of Total Sales')
        fig = px.pie(res, values=res.sales, names=res.index,
                     height=300, width=200)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=0),)
        st.plotly_chart(fig, use_container_width=True)

    def itemSalesCount(data):
        st.write("Total Item Sales Graph")
        itemdf=pd.DataFrame(data.groupby('item', as_index=False)['sales'].sum())
        st.bar_chart(itemdf, x="item", y="sales", color="#ffaa0088")

    def itemDonut(data):
        itemdf = pd.DataFrame(data.groupby('item', as_index=False)['sales'].sum())
        fig = px.pie(itemdf, names='item', values='sales', hole=0.4)
        fig.update_layout(title='Item Donut Chart')
        st.plotly_chart(fig)

    def plotMonthlySales(data):
        st.write("Monthly Sales Graph")
        st.line_chart(data, x="date", y="sales", color="#ffaa0088")



    uploaded_file = st.file_uploader("Choose a .csv file")
    if uploaded_file is not None:
        with st.spinner("Just a moment ..."):
            # time.sleep(5)
            data = pd.read_csv(uploaded_file)
            # print(data.columns.tolist())
            if 'item' in data.columns.tolist() and 'date' in data.columns.tolist() and 'sales' in data.columns.tolist():
                try:
                    data.interpolate(method='linear', inplace=True)
                    try:
                        data['date'] = pd.to_datetime(data['date'], format="%d-%m-%Y")
                    except:
                        data['date'] = pd.to_datetime(data['date'], format="%d.%m.%Y")
                except:
                    data['date'] = pd.to_datetime(data['date'], format="%Y-%m-%d")
                    data['date'] = data['date'].dt.strftime('%d-%m-%Y')
                    data['date'] = pd.to_datetime(data['date'], format="%d-%m-%Y")

                m_df = data.groupby(data['date'].dt.strftime('%Y-%m'))['sales'].sum().reset_index()
                st.session_state['data'] = data
                st.session_state['daily_data'] = data
                st.session_state['m_df'] = m_df
                st.session_state['uploaded_file'] = uploaded_file.name
                st.success('Done!')
            else:
                st.write("Incorrect Format")
                st.write("Must include item, sales and  date column")


    if 'data' in st.session_state and 'uploaded_file' in st.session_state:
        st.write(f"Uploaded File : ",st.session_state['uploaded_file'])
        st.dataframe(st.session_state['data'].head(), use_container_width=True)
        st.markdown("<h1 style='text-align: center; color: white;'>Trends</h1>", unsafe_allow_html=True)
        plotMonthlySales(st.session_state['m_df'])
        cols = st.columns([1, 1, 1], gap="large")
        data=st.session_state['data']
        try:
            data['date'] = pd.to_datetime(data['date'], format="%d-%m-%Y")
        except:
            data['date'] = pd.to_datetime(data['date'], format="%Y-%m-%d")
            data['date'] = data['date'].dt.strftime('%d-%m-%Y')
            data['date'] = pd.to_datetime(data['date'], format="%d-%m-%Y")
        with cols[0]:
            st.markdown(f"<h1 style='text-align: center; color: white;'>{st.session_state['data']['date'].min().date()}</h1>",unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color: white;'>Start Date</h2>", unsafe_allow_html=True)

        with cols[1]:
            st.markdown(f"<h1 style='text-align: center; color: white;'>{len(st.session_state['data'].item.unique())}</h1>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color: white;'>Total Items</h2>", unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f"<h1 style='text-align: center; color: white;'>{st.session_state['data']['date'].max().date()}</h1>",
                unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color: white;'>End Date</h2>", unsafe_allow_html=True)

        st.write("")
        st.write("")
        cols = st.columns([1,1], gap="large")
        with cols[0]:
            st.markdown(
                f"<h1 style='text-align: center; color: white;'>{len(st.session_state['data'])}</h1>",
                unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color: white;'>Total Records</h2>", unsafe_allow_html=True)

        with cols[1]:
            st.markdown(
                f"<h1 style='text-align: center; color: white;'>{st.session_state['data']['sales'].mean():.2f}</h1>",
                unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color: white;'>Mean S/D</h2>", unsafe_allow_html=True)
        st.write("")
        st.write("")
        itemSalesCount(st.session_state['data'])
        st.write("")
        st.write("")
        cols = st.columns([1, 1], gap="large")
        with cols[0]:
            itemDonut(st.session_state['data'])
        with cols[1]:
            Pie(data)
        st.write("")
        st.write("")

if selected == "Product":


    def itemMonthlySales(data):
        with st.spinner("Just a moment ..."):
            st.write("Monthly Item Sales Graph")
            st.bar_chart(data, x="date", y="sales", color="#ffaa0088")


    def Pie(data,option):
        with st.spinner("Just a moment ..."):
            data['date'] = pd.to_datetime(data['date'], format="%d-%m-%Y")
            res = pd.DataFrame(data.groupby(data['date'].dt.strftime('%B'))['sales'].sum())
            # print(res)
            yrs = pd.DataFrame(data.groupby(data['date'].dt.strftime('%Y'))['sales'].sum())
            yrs['date']=yrs.index

            cols = st.columns([1, 1],gap="large")

            with cols[0]:

                fig = px.pie(res, values=res.sales, names=res.index,
                             title=f'Monthly Distribution of Sales of Item : {option}',
                             height=300, width=200)
                fig.update_layout(margin=dict(l=20, r=20, t=30, b=0), )
                st.plotly_chart(fig, use_container_width=True)

            with cols[1]:
                # st.line_chart(yrs, x="date", y="sales", color="#ffaa0088")
                st.write(f"Yearly Sales of Item : {option}")
                st.bar_chart(yrs, x="date", y="sales", color="#ffaa0088")
                # st.write("2")



    if 'data' in st.session_state:
        try:
            with st.spinner("Just a moment ..."):
                st.write(f"Uploaded File : ", st.session_state['uploaded_file'])
                un = st.session_state['data'].item.unique().tolist()

                option = st.selectbox('Select Item ID : ', un)
                st.write('Product selected:', option)
                st.session_state['option']=option
            with st.spinner("Just a moment ..."):
                st.markdown("<h1 style='text-align: center; color: white;'>Item Trends</h1>", unsafe_allow_html=True)
                d = st.session_state['data']

                try:
                    d['date'] = pd.to_datetime(d['date'], format="%d-%m-%Y")
                except:
                    d['date'] = pd.to_datetime(d['date'], format="%Y-%m-%d")
                    d['date'] = d['date'].dt.strftime('%d-%m-%Y')
                    d['date'] = pd.to_datetime(d['date'], format="%d-%m-%Y")
                itemMonthly = prediction.PredictionModel.itemDF(d, option)
                itemMonthlySales(itemMonthly)
                st.write("")
                st.write("")

                Pie(itemMonthly, option)
        except:
            st.title("Something went wrong!\nPlease reload the page.")
    else:
        st.write(" ")
        selected='Home'
        streamlit_js_eval(js_expressions="parent.window.location.reload()")
        mycode = "<script>alert('Upload a .csv file in home page')</script>"
        components.html(mycode, height=0, width=0)
        st.toast('Upload a .csv file in home page', icon='⚠')
if selected == "Predict":

    if 'data' in st.session_state and 'option' in st.session_state:
        try:
            st.write("")
            with st.spinner("Just a moment ..."):
                st.write(f"Uploaded File : {st.session_state['uploaded_file']}")
                st.write(f"Selected Product : {st.session_state['option']}")
                d=st.session_state['data']
                d['date'] = pd.to_datetime(d['date'], format="%Y-%m-%d")
                d['date'] = d['date'].dt.strftime('%d-%m-%Y')
                resdf = prediction.PredictionModel.runModel(d, st.session_state['option'])
                og_resdf=resdf.copy()
                # print(resdf)
                resdf['date'] = pd.to_datetime(resdf['date'], format="%Y-%m-%d")
                resdf['date'] = resdf['date'].dt.strftime('%B %Y')

                st.dataframe(resdf, use_container_width=True)
                st.write()
                st.markdown(
                    f"<h1 style='text-align: center; color: white;'>Total Predicted Sales Next Year: {round(resdf['pred_value'].sum())}</h1>",
                    unsafe_allow_html=True)
                st.write()
                resdf['date'] = pd.to_datetime(resdf['date'])
                resdf = resdf.sort_values(by='date')
                st.line_chart(
                    resdf, x="date", y="pred_value", color="#ffaa0088"
                )
                st.image(f'model_output/forecasting.png', caption='Prediction')
        except:
            st.title("Something went wrong!\nPlease reload the page.")

    else:
        st.write(" ")
        selected='Home'
        streamlit_js_eval(js_expressions="parent.window.location.reload()")
        mycode = "<script>alert('Upload a .csv file in home page')</script>"
        components.html(mycode, height=0, width=0)
        st.toast('Upload a .csv file in home page', icon='⚠')
