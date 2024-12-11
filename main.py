from methods import *
#web
st.title("Автоматизация цепочек поставок")

if 'store_number' not in st.session_state:
    st.session_state.store_number = 1
if 'today_date' not in st.session_state:
    st.session_state.today_date = datetime.today()
if 'forecast_date' not in st.session_state:
    st.session_state.forecast_date = datetime.today()

# Загрузка CSV файлов
expected_columns_sales = {
    "Store": "int64",
    "Date": "object",
    "Weekly_Sales": "float64"
}
uploaded_file = st.file_uploader("Выберите CSV файл с историей спроса", type="csv")
if uploaded_file is not None:
    df_original = pd.read_csv(uploaded_file)
    if not validate_dataframe(df_original, expected_columns_sales):
        st.error("Файл не соответствует ожидаемому формату.")
        reset_state()
        st.stop()
else:
    st.stop()

df_original['Date'] = pd.to_datetime(df_original['Date'])

unique_stores = df_original['Store'].unique()
for store in unique_stores:
    store_df = df_original[df_original['Store'] == store]
    date_1=store_df.iloc[0]['Date']
    for i in range(1,len(store_df)):
        date_2=store_df.iloc[i]['Date']
        if (date_2-date_1).days!=7:
            st.error("Данные файла некорректны.")
            st.stop()
        date_1=date_2

test_df = pd.DataFrame()
st.write(df_original.head())

store_number = st.number_input("Введите номер магазина", min_value=1, step=1, value=st.session_state.store_number)
if store_number not in df_original['Store'].values:
    st.error(f"Магазин с номером {store_number} не найден в данных.")
    st.stop()

today_date = st.date_input("Текущая дата", value=st.session_state.today_date)
forecast_date = st.date_input("Дата прогноза", value=st.session_state.forecast_date)
today_date = pd.to_datetime(today_date)
forecast_date=pd.to_datetime(forecast_date)

if forecast_date <= today_date:
    st.error("Дата прогноза должна быть позже текущей даты.")
    st.stop()

flag, df = process_store_data(df_original, store_number, today_date)
if not flag:
    st.error("Предоставленные вами данные не соответствуют выбранным датам.")
    st.stop()

store_df = df[df['Store'] == store_number]
start_date = store_df.iloc[134]['Date']
delta_days = (forecast_date - start_date).days
delta_weeks = delta_days // 7+31
if delta_days % 7 != 0:
    delta_weeks += 1
test_df = store_df.iloc[135:]
if len(test_df)>delta_weeks:
    test_df=test_df.iloc[:delta_weeks]
st.write(f"Количество недель для прогноза: {delta_weeks-31}")
if delta_weeks > 52:
    st.error(f"Выберете более раннюю дату для прогноза.")
    st.stop()

alpha_final, beta_final, gamma_final = hyperparm_selection(df)
sales_matrix = store_df.pivot(index='Store', columns='Date', values='Weekly_Sales').to_numpy()
model = ExpSmooth(sales_matrix[0], alpha_final, beta_final, gamma_final, delta_weeks)
model.smoothing()
results = model.result[31:]
compare_results = results[:len(test_df)]

required_goods=round(results[-1],2)

# Опции для построения графиков
if len(test_df)>1:
    with st.expander("Построить график сравнения с реальными данными"):
        make_graphic_comparison(compare_results,test_df)

# Вычисление и отображение метрик
if len(test_df)>1:
    with st.expander("Рассчитать метрики"):
        r2, mape, rmse = metrics(test_df, compare_results, len(compare_results))
        r2 = round(r2, 2)
        mape = round(mape, 2)
        rmse = round(rmse, 2)
        st.write(f"R2 Score: {r2}")
        st.write(f"MAPE: {mape}%")
        st.write(f"RMSE: {rmse}")

with st.expander("Построить график прогноза"):
    make_graphic(start_date, delta_weeks-31, results)

# Загрузка CSV-файлов для данных о поставщиках и доставке
expected_columns_sup = {
    "Manufacturer Name": "object",
    "Quality M": "int64",
    "Material price": "int64",
    "Information sharing": "int64",
    "After sales service": "int64",
    "Lead time": "int64",
    "Quantity discount": "int64",
    "Occupational health and safety system M": "int64",
    "Production time": "int64"
}
expected_columns_del = {
    "Delivery Company Name": "object",
    "Quality D": "int64",
    "Delivery on time": "int64",
    "Occupational health and safety system D": "int64",
    "Transportation cost": "int64",
    "Delivery time": "float64"
}
uploaded_suppliers_file = st.file_uploader("Выберите CSV файл с данными о производителях", type="csv")
if uploaded_suppliers_file is not None:
    suppliers_df = pd.read_csv(uploaded_suppliers_file)
    if not validate_dataframe(suppliers_df, expected_columns_sup):
        st.error("Файл не соответствует ожидаемому формату.")
        st.stop()
else:
    st.stop()
st.write(suppliers_df.head())

uploaded_delivery_file = st.file_uploader("Выберите CSV файл с данными логистических компаний", type="csv")
if uploaded_delivery_file is not None:
    delivery_df = pd.read_csv(uploaded_delivery_file)
    if not validate_dataframe(delivery_df, expected_columns_del):
        st.error("Файл не соответствует ожидаемому формату.")
        st.stop()
else:
    st.stop()
st.write(delivery_df.head())

combined_df = suppliers_df.merge(delivery_df, how='cross')

# Выгрузка модели из бд
random_forest_model = load_model_from_db()

# Классификация записей
excellent_suppliers = []
satisfactory_suppliers = []

for index, row in combined_df.iterrows():
    prediction = random_forest_model.predict(row.to_dict())
    if prediction == 'excellent':
        excellent_suppliers.append(row)
    elif prediction == 'satisfactory':
        satisfactory_suppliers.append(row)
excellent_suppliers_df = pd.DataFrame(excellent_suppliers).dropna()
satisfactory_suppliers_df = pd.DataFrame(satisfactory_suppliers).dropna()

# Вывод результатов
lines=[]
if not excellent_suppliers_df.empty:
    unique_manufacturers = sorted(excellent_suppliers_df['Manufacturer Name'].unique(), key=extract_number)
    with st.expander("Лучшие производители:"):
        for manufacturer in unique_manufacturers:
            st.write(manufacturer)
    unique_deliveries = sorted(excellent_suppliers_df['Delivery Company Name'].unique(), key=extract_number)
    with st.expander("Лучшие логистические компании:"):
        for deliver in unique_deliveries:
            st.write(deliver)
    with st.expander("Поставщики с оценкой 'excellent'"):
        line=f"Товар в количестве: {required_goods}"
        st.write(line)
        lines.append(line)
        for i in range(len(excellent_suppliers_df)):
            supplier=excellent_suppliers_df.iloc[i]
            order_date = forecast_date - timedelta(days=(supplier['Production time'] + supplier['Delivery time']))
            if order_date > today_date:
                order_date = order_date.strftime('%d.%m.%Y')
                line = f"Можно заказать у поставщика: {supplier['Manufacturer Name']} с доставкой компанией: {supplier['Delivery Company Name']} до: {order_date}"
                st.write(line)
                lines.append(line)
elif not satisfactory_suppliers_df.empty:
    unique_manufacturers = sorted(satisfactory_suppliers_df['Manufacturer Name'].unique(), key=extract_number)
    with st.expander("Производители средней категории:"):
        for manufacturer in unique_manufacturers:
            st.write(manufacturer)
    unique_deliveries = sorted(satisfactory_suppliers_df['Delivery Company Name'].unique(), key=extract_number)
    with st.expander("Логистические компании средней категории:"):
        for deliver in unique_deliveries:
            st.write(deliver)
    with st.expander("Поставщики с оценкой 'satisfactory'"):
        line=f"Товар в количестве: {required_goods}"
        st.write(line)
        lines.append(line)
        for i in range(len(satisfactory_suppliers_df)):
            supplier=satisfactory_suppliers_df.iloc[i]
            order_date = forecast_date - timedelta(days=(supplier['Production time'] + supplier['Delivery time']))
            if order_date > today_date:
                order_date = order_date.strftime('%d.%m.%Y')
                line=f"Можно заказать у поставщика: {supplier['Manufacturer Name']} с доставкой компанией: {supplier['Delivery Company Name']} до: {order_date}"
                st.write(line)
                lines.append(line)
else:
    st.write("К сожалению, нет доступных поставщиков с необходимыми критериями.")
if lines:
    pdf_file = generate_pdf(lines)
    st.download_button("Скачать информацию о поставщиках", pdf_file, "supplier_info.pdf", "application/pdf")