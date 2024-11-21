import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pydeck as pdk
import datetime
import os

from keboola_streamlit import KeboolaStreamlit
from io import BytesIO
from PIL import Image
from tempfile import gettempdir
from openai import OpenAI
from kbcstorage.client import Client, Files
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")

ASSISTANT_ID='asst_ZcyM7WhcXFM7Et6FqFwpP0aX'
FILE_ID='file-lwe2KavkmH14XzCd6K1YcNHG'
MINI_LOGO_URL = 'https://components.keboola.com/images/default-app-icon.png'
LOGO_URL = 'https://assets-global.website-files.com/5e21dc6f4c5acf29c35bb32c/5e21e66410e34945f7f25add_Keboola_logo.svg'

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
kbc_client = Client(st.secrets['kbc_url'], st.secrets['KEBOOLA_TOKEN'])

@st.cache_data(show_spinner='Loading data...🍟🍔🧋')
def read_data(table_name):
    keboola = KeboolaStreamlit(st.secrets['kbc_url'], st.secrets['KEBOOLA_TOKEN'])
    df = keboola.read_table(table_name)
    return df

def write_table(table_id: str, df: pd.DataFrame, is_incremental: bool = False):    
    csv_path = f'{table_id}.csv'
    try:
        df.to_csv(csv_path, index=False)
        
        files = Files(st.secrets['kbc_url'], st.secrets['KEBOOLA_TOKEN'])
        file_id = files.upload_file(file_path=csv_path, tags=['file-import'],
                                    do_notify=False, is_public=False)
        job = kbc_client.tables.load_raw(table_id=table_id, data_file_id=file_id, is_incremental=is_incremental)
        
    except Exception as e:
        st.error(f'Data upload failed with: {str(e)}')
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)
    return job

def generate_response(prompt):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred during content generation. Please try again.")
        return ''
    
def sentiment_color(val):
    color_map = {
        'Positive': 'color: #34A853',
        'Mixed': 'color: #FBBC05',
        'Negative': 'color: #EA4335',
        'Unknown': 'color: #B3B3B3'
    }
    return color_map.get(val, '')

@st.fragment
def generate_top_x(top_x, data, num_reviews):
    def calculate_rating_distribution(ratings):
        """Calculate the normalized percentage of each rating."""
        rating_counts = pd.Series(ratings).value_counts(normalize=True).sort_index()
        return rating_counts
    
    col1, col2 = st.columns(2, gap='small')
    top_locations = data[data['COUNT'] >= num_reviews].head(top_x)
    top_rating_distribution = top_locations['RATING'].apply(calculate_rating_distribution).fillna(0)
    top_rating_distribution.index = top_locations.apply(lambda row: f"{row['ADDRESS']}", axis=1)

    top_rating_distribution = top_rating_distribution.sort_index(axis=1, ascending=False)
    top_rating_distribution_reversed = top_rating_distribution.iloc[::-1]

    fig_top = px.bar(
        top_rating_distribution_reversed,
        x=top_rating_distribution_reversed.columns,
        y=top_rating_distribution_reversed.index,
        orientation='h',
        labels={'value': 'Percentage', 'index': 'Location', 'rating': 'Rating', 'variable': 'Rating'},
        title=f'Rating Distribution for Top {top_x} Locations',
        color_discrete_map=rating_colors_index
    )
    fig_top.update_layout(
        showlegend=False, 
        xaxis_title=None, 
        yaxis_title=None, 
        xaxis_tickformat='.0%',
        xaxis={'showticklabels': False},
        yaxis={'tickvals': top_rating_distribution_reversed.index, 'ticktext': top_rating_distribution_reversed.index}
    )
    col1.plotly_chart(fig_top, use_container_width=True)
    
    bottom_locations = data[data['COUNT'] >= num_reviews].tail(top_x)
    bottom_rating_distribution = bottom_locations['RATING'].apply(calculate_rating_distribution).fillna(0)
    bottom_rating_distribution.index = bottom_locations.apply(lambda row: f"{row['ADDRESS']}", axis=1)

    bottom_rating_distribution = bottom_rating_distribution.sort_index(axis=1, ascending=False)
    fig_bottom = px.bar(
        bottom_rating_distribution,
        x=bottom_rating_distribution.columns,
        y=bottom_rating_distribution.index,
        orientation='h',
        labels={'value': 'Percentage', 'index': 'Location', 'rating': 'Rating', 'variable': 'Rating'},
        title=f'Rating Distribution for Bottom {top_x} Locations',
        color_discrete_map=rating_colors_index
    )
    fig_bottom.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None, xaxis_tickformat='.0%', xaxis={'showticklabels': False}, yaxis={'tickvals': bottom_rating_distribution.index, 'ticktext': bottom_rating_distribution.index})
    col2.plotly_chart(fig_bottom, use_container_width=True)

@st.fragment
def generate_rating_chart(data):
    rating_counts = data['RATING'].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)

    rating_chart_data = pd.DataFrame({
        'Rating': rating_counts.index,
        'Count': rating_counts.values,
        'Color': [rating_colors[rating] for rating in rating_counts.index]
    })
    
    fig_ratings = px.bar(
        rating_chart_data,
        x='Count',
        y='Rating',
        orientation='h',
        labels={'Count': 'Count', 'Rating': 'Rating'},
        title='Count of Ratings',
        text='Count', 
        color='Color',  
        color_discrete_map='identity'  
    )

    fig_ratings.update_traces(textposition='inside', textfont_color='white', texttemplate='%{text:,}')
    fig_ratings.update_layout(xaxis_title=None)
    fig_ratings.update_yaxes(title_text=None)

    st.plotly_chart(fig_ratings, use_container_width=True)

def generate_html(emoji, label, main_value, sub_label, sub_value, always_show_subtext=False):
    if main_value != sub_value:
        subtext = f"{sub_label}: {sub_value}"
    elif always_show_subtext:
        subtext = f"{sub_label}: {sub_value}"
    else:
        subtext = sub_label 

    html_code = f"""
    <div style='text-align: center; color: black;'>
        <h1 style='font-size: 16px;'>{emoji} {label}</h1>
        <span style='font-size: 38px;'>{main_value}</span>
        <p style='margin:0; color: gray; font-size: 0.8em;'>{subtext}</p>
        <br>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

def metrics(all_data, filtered_data, show_pie=False):    
    # Metrics for all
    all_review_count = len(all_data)
    all_avg_rating = all_data['RATING'].mean() if all_review_count > 0 else 0
    all_unique_locations = all_data['PLACE_ID'].nunique()
    
    # Metrics for filtered
    filtered_review_count = len(filtered_data)
    filtered_avg_rating = filtered_data['RATING'].mean() if all_review_count > 0 else 0
    filtered_unique_locations = filtered_data['PLACE_ID'].nunique()

    col1, col2, col3 = st.columns(3)
    
    with col1:
        generate_html('📝', "Reviews", f"{filtered_review_count:,}", "out of", f"{all_review_count:,}", always_show_subtext=True)
    with col2: 
        if show_pie:
            html_code = """
                <div style='text-align: center; color: black;'>
                    <h1 style='font-size: 16px;'>😎 Sentiment Distribution</h1>
                </div>
                """

            st.markdown(html_code, unsafe_allow_html=True)
            sentiment_counts = filtered_data['OVERALL_SENTIMENT'].value_counts()
            percentages = (sentiment_counts / sentiment_counts.sum() * 100).round(2)
            fig_sentiment_donut = px.pie(
                sentiment_counts,
                values=sentiment_counts.values,
                names=[f"{count:,} ({percentage:.1f}%)" for count, percentage in zip(sentiment_counts.values, percentages)],
                hole=0.3,
                color=sentiment_counts.index,
                color_discrete_map=word_rating_colors
            )

            fig_sentiment_donut.update_traces(
                textinfo='none',
                hoverinfo='skip'
            )
            
            fig_sentiment_donut.update_layout(
                height=120,
                margin=dict(l=20, r=20, t=0, b=50),  
                paper_bgcolor='rgba(0,0,0,0)',         
                showlegend=True,                       
                legend=dict(
                    orientation="h",                   
                    x=0.5,                             
                    xanchor="center",
                    y=-0.2,
                    yanchor="top"
                ),
                hovermode=False
            )
            st.plotly_chart(fig_sentiment_donut, use_container_width=True)
        else:
            generate_html('⭐️', 'Average Rating', f"{filtered_avg_rating:.2f}", "avg for all", f"{all_avg_rating:.2f}")
    with col3:
        generate_html('📍', 'Locations', f"{filtered_unique_locations:,}", "out of", f"{all_unique_locations:,}", always_show_subtext=True)

def clean_list_string(x):
    if pd.isna(x):
        return ''
    try:
        # Remove brackets, quotes, and split by comma
        cleaned = x.strip('[]').replace('"', '').replace("'", '')
        # Join items with comma and space
        return ', '.join(cleaned.split(','))
    except:
        return ''

# Create and display network graph of entities and attributes
def create_network_graph(attributes, slider_entities):
    # Get top entities by total attribute counts
    pivot_attrs = attributes.pivot(index='entity', columns='attribute', values='count').fillna(0)
    pivot_attrs['Total'] = pivot_attrs.sum(axis=1)
    top_entities = pivot_attrs.nlargest(slider_entities, 'Total').index.tolist()
    
    # Initialize graph and figure
    G = nx.Graph()
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    
    # Calculate entity positions in a circle
    entity_positions = calculate_entity_positions(top_entities)
    
    # Add nodes and edges to graph
    add_nodes_and_edges(G, top_entities, attributes, entity_positions)
    
    # Position attribute nodes
    pos = position_attribute_nodes(G, entity_positions)
    
    # Draw the network
    draw_network(G, pos, top_entities)
    
    # Configure plot
    ax.axis('off')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    
    return fig

def calculate_entity_positions(entities, radius=1.5):
    positions = {}
    num_entities = len(entities)
    for i, entity in enumerate(entities):
        angle = 2 * np.pi * i / num_entities
        positions[entity] = (radius * np.cos(angle), radius * np.sin(angle))
    return positions

def add_nodes_and_edges(G, top_entities, attributes, entity_positions):
    for entity in top_entities:
        G.add_node(entity, node_type='entity')
        entity_attrs = attributes[attributes['entity'] == entity]
        
        for _, row in entity_attrs.iterrows():
            attr, count = row['attribute'], row['count']
            if count > 0:
                if attr not in G.nodes():
                    G.add_node(attr, node_type='attribute')
                G.add_edge(entity, attr, weight=count)

def position_attribute_nodes(G, entity_positions, scale_factor=0.8):  # Increased scale factor
    pos = entity_positions.copy()
    attr_nodes = [n for n in G.nodes() if n not in entity_positions]

    # Count how many entities each attribute is connected to
    attr_connections = {}
    for attr in attr_nodes:
        connected_entities = [n for n in G.neighbors(attr) if n in entity_positions]
        attr_connections[attr] = len(connected_entities)

    # Sort attributes by number of connections
    attr_nodes = sorted(attr_nodes, key=lambda x: attr_connections[x], reverse=True)
    
    # Keep track of occupied positions
    occupied_positions = []
    min_distance = 0.2  # Reduced minimum distance between nodes

    for attr in attr_nodes:
        connected_entities = [n for n in G.neighbors(attr) if n in entity_positions]
        if connected_entities:
            attempts = 0
            while attempts < 50:
                if attr_connections[attr] > 1:
                    # Place multi-connected attributes closer to their connected entities
                    x = sum(entity_positions[e][0] for e in connected_entities) / len(connected_entities)
                    y = sum(entity_positions[e][1] for e in connected_entities) / len(connected_entities)
                    x *= scale_factor
                    y *= scale_factor
                    
                    # Reduced offset values
                    base_offset = 0.15  # Smaller base offset
                    offset = base_offset + (0.1 * attempts / 50)  # Smaller offset range
                    angle = np.random.uniform(0, 2*np.pi)
                    x += offset * np.cos(angle)
                    y += offset * np.sin(angle)
                else:
                    # Single-connection attributes closer to their entity
                    entity = connected_entities[0]
                    angle = 2 * np.pi * attempts / 50
                    base_radius = 0.25  # Reduced base radius
                    radius = base_radius + (0.1 * attempts / 50)  # Smaller radius range
                    x = entity_positions[entity][0] + radius * np.cos(angle)
                    y = entity_positions[entity][1] + radius * np.sin(angle)

                position = (x, y)
                if all(np.sqrt((x - ox)**2 + (y - oy)**2) > min_distance 
                        for ox, oy in occupied_positions):
                    occupied_positions.append(position)
                    pos[attr] = position
                    break
                
                attempts += 1
            
            if attr not in pos:
                pos[attr] = position
                occupied_positions.append(position)

    return pos

def draw_network(G, pos, top_entities):
    # Draw nodes
    attr_nodes = [n for n in G.nodes() if n not in top_entities]
    nx.draw_networkx_nodes(G, pos, nodelist=top_entities, node_color='#238dff',
                            node_size=2500, node_shape='o')
    nx.draw_networkx_nodes(G, pos, nodelist=attr_nodes, node_color='#e6f2ff',
                            node_size=1000, alpha=0.7, node_shape='o')
    
    # Draw edges with different colors for each entity
    colors = plt.cm.rainbow(np.linspace(0, 1, len(top_entities)))
    for i, entity in enumerate(top_entities):
        # Get edges connected to this entity
        entity_edges = [(u,v) for (u,v) in G.edges() if u == entity or v == entity]
        if entity_edges:
            edge_weights = [G[u][v]['weight'] for u, v in entity_edges]
            nx.draw_networkx_edges(G, pos, edgelist=entity_edges, 
                                    width=[w/max(edge_weights)*2 for w in edge_weights],
                                    edge_color=[colors[i]], alpha=0.7)
    
    # Draw labels
    entity_labels = {node: node for node in top_entities}
    attr_labels = {node: node for node in attr_nodes}
    
    # Use original positions for labels to place them in the center of nodes
    nx.draw_networkx_labels(G, pos, labels=entity_labels, font_size=10, font_color='white', font_weight='600')
    nx.draw_networkx_labels(G, pos, labels=attr_labels, font_size=8)

@st.fragment
def display_network_graph(attributes):
    st.markdown("##### Entity-Attribute Relations")
    st.caption("_See up to top 20 mentioned entities and their attributes._")
    col1, col2 = st.columns([0.9, 0.1], vertical_alignment='center')
    num_entities = col2.number_input("Select the number of entities", min_value=1, max_value=20, value=5)
    fig = create_network_graph(attributes, num_entities)
    col1.pyplot(fig, use_container_width=True)


if 'thread_id' not in st.session_state:
    st.session_state.thread_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', 'content': 'Welcome! How can I assist you today?'}]
if 'table_written' not in st.session_state:
    st.session_state.table_written = False
if 'new_prompt' not in st.session_state:
    st.session_state.new_prompt = None
if 'instruction' not in st.session_state:
    st.session_state.instruction = ''
if 'regenerate_clicked' not in st.session_state:
    st.session_state.regenerate_clicked = False

options = ['About', 'Locations', 'Overview', 'AI Analysis', 'Support', 'Assistant'] #'Sales Performance', 'P-Mix']
icons=['info-circle', 'pin-map-fill', 'people', 'file-bar-graph', 'chat-heart', 'robot'] #'currency-exchange', 'menu-up']

menu_id = option_menu(None, options=options, icons=icons, key='menu_id', orientation="horizontal")

LOGO_URL = 'https://assets-global.website-files.com/5e21dc6f4c5acf29c35bb32c/5e21e66410e34945f7f25add_Keboola_logo.svg'

st.sidebar.markdown(
    f'''
        <div style="text-align: center; margin-top: 20px; margin-bottom: 40px;">
            <img src="{LOGO_URL}" alt="Logo" width="200">
        </div>
    ''',
    unsafe_allow_html=True
)


data = read_data('out.c-257-data-model.MERGED_DATA_FINAL') #pd.read_csv('review_data_cleaned.csv') #('/data/in/tables/location_review.csv')
sentences = read_data('out.c-257-data-model.REVIEW_SENTENCE')

# Read and process attribute data
attributes = pd.read_csv('relations.csv') #'/data/in/tables/relations.csv')
attributes['entity'] = attributes['entity'].replace('burgers', 'burger')
pronouns_to_remove = ['i', 'you', 'she', 'he', 'it', 'we', 'they', 'I', 'You', 'She', 'He', 'It', 'We', 'They']
attributes = attributes[~attributes['entity'].isin(pronouns_to_remove)]
attributes = attributes.groupby(['entity', 'attribute'])['count'].sum().reset_index()
attributes = attributes[attributes['count'] > 2]

review_data = data.copy()

review_data['CATEGORY'] = review_data['CATEGORY'].apply(lambda x: x.split('|') if pd.notna(x) else [])
review_data['CATEGORY_GROUP'] = review_data['CATEGORY_GROUP'].apply(lambda x: x.split('|') if pd.notna(x) else [])
review_data['TOPIC'] = review_data['TOPIC'].apply(lambda x: x.split('|') if pd.notna(x) else [])
review_data['ENTITIES'] = review_data['ENTITIES'].apply(lambda x: x.split('|') if pd.notna(x) else [])

review_data['REVIEW_DATE'] = pd.to_datetime(review_data['REVIEW_DATE']).dt.date
review_data['RATING'] = review_data['RATING'].astype(int)

rating_colors = {0: '#B3B3B3', 1: '#EA4335', 2: '#e98f41', 3: '#FBBC05', 4: '#a5c553', 5: '#34A853'}
rating_colors_index = {'0': '#B3B3B3', '1': '#EA4335', '2': '#e98f41', '3': '#FBBC05', '4': '#a5c553', '5': '#34A853'}
word_rating_colors = {'Negative': '#EA4335', 'Mixed': '#FBBC05', 'Unknown': '#B3B3B3', 'Positive': '#34A853'} 

if menu_id == 'About':
    st.title('Keboola QSR 360 Solution')
        #  color: #208dff
    st.write("Own your data, automate workflows, and drive smarter decisions across all restaurant initiatives.")
    col1, col2 = st.columns([0.4, 0.6], gap='large', vertical_alignment='center')

    col1.markdown("<strong>Keboola’s QSR 360</strong> is revolutionizing the franchise business by seamlessly integrating data, automating workflows, and providing advanced analytics for enhanced operational efficiency and strategic decision-making.", unsafe_allow_html=True)
    col2.image("https://i.ibb.co/jysDwBP/66bdc321b8987869a60f9d1b-QSR-diagram-2-2.jpg")
    st.divider()
    st.subheader('Demo: Customer Voice and AI Support')
    st.markdown("""
        Use powerful AI analysis to improve service quality and customer satisfaction. Analyzing feedback helps identify areas for improvement, ensuring high service standards.
        \n\nThe demo environment includes the following sections:
        \n\n1. The **Overview** tab provides instant, out-of-the-box insights into online reviews for all your restaurant locations. It delivers a comprehensive overview of general ratings, key trends, and identifies any anomalies in customer feedback. The app's easy-to-use interface allows teams to quickly assess the overall sentiment across various locations, helping them stay on top of evolving customer perceptions.
        \n\n2. The **AI Analysis** tab is a powerful tool designed to analyze customer reviews by utilizing advanced AI-driven sentiment analysis. The app categorizes feedback into key areas, breaking down each category into subgroups and specific topics to provide deep insights into customer sentiment. This structured approach helps businesses understand detailed feedback patterns and address specific areas of concern or praise.
        \n\nHere’s an example of how the app organizes the analysis:
        \n\n- **Category:** Broad themes identified in customer reviews (e.g., Food, People, Experience)
        \n\n- **Group:** More specific aspects of the category (e.g., Quality, Team, Ordering)
        \n\n- **Topics:** Fine-grained details mentioned by customers (e.g., Taste, Hospitality, Pricing Accuracy)
        \n\nThis in-depth analysis allows teams to pinpoint specific trends in customer sentiment, improving operational decision-making and enhancing overall customer satisfaction by responding to the most critical feedback efficiently.
        \n\n3. The **Support** tab is an interactive tool designed to assist Customer Support teams in efficiently managing guest communication through AI-generated review replies. These AI-crafted responses are carefully reviewed and approved by the team, ensuring quality and personalization. Using a funnel or mailbox-style interface, the app facilitates team collaboration, enabling swift responses to both positive and negative feedback. By streamlining this process, the app helps boost guest satisfaction and loyalty, ensuring timely and thoughtful engagement with all customer reviews.
    """, unsafe_allow_html=True)
    
# Brand Selection
brand_options = review_data['BRAND'].unique().tolist()    
brand = st.sidebar.multiselect('Select a brand', brand_options, brand_options[0], placeholder='All') # brand??
if len(brand) > 0:
    selected_brand = brand
else:
    selected_brand = brand_options

# State Selection
state_options = sorted(review_data['STATE'].unique().tolist())
state = st.sidebar.multiselect('Select a state', state_options, placeholder='All')
if len(state) > 0:
    selected_state = state
else:
    selected_state = state_options

# City Selection
city_options = sorted(review_data[review_data['STATE'].isin(selected_state)]['CITY'].unique().tolist())
city = st.sidebar.multiselect('Select a city', city_options, placeholder='All')
if len(city) > 0:
    selected_city = city
    location_options = sorted(review_data[review_data['CITY'].isin(selected_city)]['ADDRESS'].unique().tolist())
else:
    selected_city = city_options
    location_options = sorted(review_data[review_data['STATE'].isin(selected_state)]['ADDRESS'].unique().tolist())

# Location Selection
location = st.sidebar.multiselect('Select a location', location_options, placeholder='All')
if len(location) > 0:
    selected_location = location
else:
    selected_location = location_options

# Sentiment Selection
sentiment_options = sorted(review_data['OVERALL_SENTIMENT'].unique().tolist())
sentiment = st.sidebar.multiselect('Select a sentiment', sentiment_options, placeholder='All')
if len(sentiment) > 0:
    selected_sentiment = sentiment
else:
    selected_sentiment = sentiment_options

# Rating Selection
rating_options = sorted(review_data['RATING'].unique().tolist())
rating = st.sidebar.multiselect('Select a review rating', rating_options, placeholder='All')
if len(rating) > 0:
    selected_rating = rating
else:
    selected_rating = rating_options

# Date Selection
date_options = ['Last Week', 'Last Month', 'Last Quarter', 'Other']
date_selection = st.sidebar.selectbox('Select a date', date_options, index=None, placeholder='All')
min_date = pd.to_datetime(review_data['REVIEW_DATE'].min()).date()
max_date = pd.to_datetime(review_data['REVIEW_DATE'].max()).date()

if date_selection is None:
    start_date = min_date
    end_date = max_date
elif date_selection == 'Other':
    start_date, end_date = st.sidebar.date_input('Select date range', value=[min_date, max_date], min_value=min_date, max_value=max_date, key='date_input')
else:
    end_date = pd.to_datetime('today').date()
    if date_selection == 'Last Week':
        start_date = (pd.to_datetime('today') - pd.DateOffset(weeks=1)).date()
    elif date_selection == 'Last Month':
        start_date = (pd.to_datetime('today') - pd.DateOffset(months=1)).date()
    elif date_selection == 'Last Quarter':
        start_date = (pd.to_datetime('today') - pd.DateOffset(months=3)).date()
selected_date_range = (start_date, end_date)

st.sidebar.divider()
st.sidebar.caption(f"The last data collection was completed on: {review_data['REVIEW_DATE'].max()}")

filtered_brand_review_data = review_data[review_data['BRAND'].isin(selected_brand)]

filtered_review_data = filtered_brand_review_data[filtered_brand_review_data['STATE'].isin(selected_state)]
filtered_review_data = filtered_review_data[filtered_review_data['CITY'].isin(selected_city)]
filtered_review_data = filtered_review_data[filtered_review_data['ADDRESS'].isin(selected_location)]
filtered_review_data = filtered_review_data[filtered_review_data['RATING'].isin(selected_rating)]
filtered_review_data = filtered_review_data[filtered_review_data['OVERALL_SENTIMENT'].isin(selected_sentiment)]
filtered_review_data = filtered_review_data[filtered_review_data['REVIEW_DATE'].between(selected_date_range[0], selected_date_range[1])]

if filtered_review_data.empty:
    st.info('No data available for the selected filters.', icon=':material/info:')
    st.stop()

if menu_id == 'Locations':
    col1, col2 = st.columns([0.3, 0.7], vertical_alignment='center')
    col1.markdown("##### Map of Locations")
    with col1:
        st.caption("Filter by average rating range")
        rating_range = st.slider(
            "Select avg rating range",
            min_value=1.0,
            max_value=5.0,
            value=(1.0, 5.0),
            step=0.1,
            label_visibility='collapsed'
        )
    
    map_data = filtered_brand_review_data.groupby(['ADDRESS', 'LATITUDE', 'LONGITUDE']).agg({
        'REVIEW_ID': 'count',
        'RATING': 'mean'
    }).reset_index()
    map_data['RATING'] = map_data['RATING'].round(2)
    
    # Filter map data based on rating range
    map_data = map_data[
        (map_data['RATING'] >= rating_range[0]) & 
        (map_data['RATING'] <= rating_range[1])
    ]
    
    def get_color(rating):
        if rating <= 2.5:
            return [234, 67, 53, 255]  # Red (#EA4335)
        elif rating <= 3.5:
            return [251, 188, 5, 255]  # Yellow (#FBBC05)
        else:
            return [52, 168, 83, 255]  # Green (#34A853)

    map_data['color'] = map_data['RATING'].apply(get_color)
    scatter_layer = pdk.Layer(
                "ColumnLayer",
                data=map_data,
                disk_resolution=12,
                radius=800,
                elevation_scale = 1000,
                get_position=["LONGITUDE", "LATITUDE"],
                get_color="color",
                get_elevation="REVIEW_ID",  # Removed brackets
                pickable=True  # Enable hover interactions
            )

    # Set initial viewport to center of Texas
    view_state = pdk.ViewState(
        latitude=29.7604,  # Approximate center of Houston
        longitude=-95.3698,  # Approximate center of Houston
        zoom=8,  # Zoom level to show most of Texas
        pitch=50
    )

    # Create the deck with updated tooltip to show average rating
    deck = pdk.Deck(
        initial_view_state=view_state,
        map_style="light",
        layers=[scatter_layer],  # Wrap layer in list
        tooltip={
            "text": "Location: {ADDRESS}\nReviews: {REVIEW_ID}\nAvg Rating: {RATING}",
            "style": {
                "backgroundColor": "white",
                "color": "black",
                "fontSize": "16px"
            }
        }
    )
    st.pydeck_chart(deck, use_container_width=True)
    st.caption("_The height of the column represents the number of reviews and the color represents the average rating._")
    
if menu_id == 'Overview':
    with st.container(border=True):
        metrics(filtered_brand_review_data, filtered_review_data)
    data_rating_sorted = (
        filtered_review_data
        .sort_values(by='REVIEW_DATE') 
        .groupby(['PLACE_ID', 'ADDRESS', 'PLACE_TOTAL_SCORE']) 
        .agg({'RATING': [lambda x: x.tolist(), 'count']})  
        .reset_index()  
        .sort_values(by=['PLACE_TOTAL_SCORE', ('RATING', 'count')], ascending=[False, False])  
    )
    data_rating_sorted.columns = ['PLACE_ID', 'ADDRESS', 'PLACE_TOTAL_SCORE', 'RATING', 'COUNT']

    col1, col2 = st.columns([0.85, 0.15], vertical_alignment='center', gap='small')
    with col2:
        st.caption("Select the number of stores")
        top_x = st.slider("Select the number of stores", min_value=1, max_value=20, value=5, label_visibility='collapsed')
        min_count = data_rating_sorted['COUNT'].min()
        max_count = data_rating_sorted['COUNT'].max()
        st.caption("Select the minimum number of reviews")   
        num_reviews = st.number_input("Select the number of reviews", min_value=min_count, max_value=max_count, value=min_count, label_visibility='collapsed')
    with col1:
        generate_top_x(top_x, data_rating_sorted, num_reviews)

    st.dataframe(
        data_rating_sorted,
        column_order=('PLACE_TOTAL_SCORE', 'ADDRESS', 'RATING', 'COUNT'),
        column_config={
        "ADDRESS": st.column_config.Column(
            "Location",
            width="large",
        ),
        "PLACE_TOTAL_SCORE": st.column_config.ProgressColumn(
            "Total Rating",
            width="small",
            help="The total rating of the location",
            format="⭐️ %.1f",
            max_value=5),
        "COUNT": st.column_config.Column("# of Ratings",
            width="small",
            help="The number of ratings for the location"
        ),
        "RATING": st.column_config.LineChartColumn(
            "Rating per Date",
            width="large",
            help="The rating during the selected date range",
            y_min=0,
            y_max=5.5,
            ),
        },
        hide_index=True, 
        use_container_width=True)
    
    col1, col2 = st.columns([0.25, 0.75], gap='medium', vertical_alignment='top')
    with col1: 
        generate_rating_chart(filtered_review_data)
    
    count_ratings_per_day = filtered_review_data.groupby(['REVIEW_DATE', 'RATING']).size().reset_index(name='COUNT')
    count_ratings_per_day['RATING'] = count_ratings_per_day['RATING'].astype(str)
    count_ratings_per_day = count_ratings_per_day.sort_values(by='RATING')  # Ensure ratings are ordered from 1 to 5

    # Calculate the average rating per day across all reviews
    average_rating_per_day = filtered_review_data.groupby('REVIEW_DATE')['RATING'].mean().reset_index()

    # Use the string-based color map
    fig_count_ratings = px.bar(
        count_ratings_per_day,
        x='REVIEW_DATE',
        y='COUNT',
        color='RATING',
        labels={'COUNT': 'Count', 'RATING': 'Rating', 'REVIEW_DATE': 'Date'},
        title='Count of Ratings and Average Rating Per Day Across All Selected Locations',
        color_discrete_map=rating_colors_index,
        opacity=0.8
    )

    # Add annotations for average rating on top of the bars
    for index, row in average_rating_per_day.iterrows():
        fig_count_ratings.add_annotation(
            x=row['REVIEW_DATE'],
            y=count_ratings_per_day[count_ratings_per_day['REVIEW_DATE'] == row['REVIEW_DATE']]['COUNT'].sum(),
            text=f"{row['RATING']:.2f}",
            showarrow=False,
            yshift=10,
            font=dict(size=10)
        )
    fig_count_ratings.update_layout(xaxis_title=None, showlegend=False)
    col2.plotly_chart(fig_count_ratings, use_container_width=True)

if menu_id == 'AI Analysis':
    with st.container(border=True): 
        metrics(filtered_brand_review_data, filtered_review_data, show_pie=True)
    
    sentiment_per_day = filtered_review_data.groupby(['REVIEW_DATE', 'OVERALL_SENTIMENT']).size().unstack()
    fig_sentiment_per_day = px.line(
        sentiment_per_day,
        x=sentiment_per_day.index,
        y=sentiment_per_day.columns,
        labels={'value': 'Count', 'REVIEW_DATE': 'Date', 'OVERALL_SENTIMENT': 'Sentiment'},
        title='Sentiment Count by Date',
        height=300,
        color_discrete_map=word_rating_colors
    )
    fig_sentiment_per_day.update_traces(mode='lines+markers')
    fig_sentiment_per_day.update_layout(xaxis_title=None, legend_title_text=None, yaxis_title=None)
    st.plotly_chart(fig_sentiment_per_day, use_container_width=True)

    avg_detailed_rating_by_date = filtered_review_data.groupby('REVIEW_DATE')[['REVIEW_DETAILED_FOOD', 'REVIEW_DETAILED_SERVICE', 'REVIEW_DETAILED_ATMOSPHERE']].mean()
    avg_detailed_rating_by_date = avg_detailed_rating_by_date.rename(columns={
        'REVIEW_DETAILED_FOOD': 'Food',
        'REVIEW_DETAILED_SERVICE': 'Service', 
        'REVIEW_DETAILED_ATMOSPHERE': 'Atmosphere'
    })
    fig_avg_detailed_rating_by_date = px.line(
        avg_detailed_rating_by_date,
        x=avg_detailed_rating_by_date.index,
        y=['Food', 'Service', 'Atmosphere'],
        labels={'x': 'Date', 'value': 'Avg Score', 'variable': 'Avg Rating', 'REVIEW_DATE': 'Date'},
        title='Average Detailed Rating by Date',
        height=300 
    )
    blue_shades = ['#57aeff', '#0a89ff', '#bddfff']
    for i, trace in enumerate(fig_avg_detailed_rating_by_date.data):
        trace.update(line=dict(color=blue_shades[i]))

    fig_avg_detailed_rating_by_date.update_traces(mode='lines+markers')
    fig_avg_detailed_rating_by_date.update_layout(xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig_avg_detailed_rating_by_date)
    
    st.divider()
    display_network_graph(attributes)
    col1, col2, col3 = st.columns([0.3, 0.35, 0.35], gap='medium', vertical_alignment='center')
    with col1:
        st.markdown("##### Classification")
        entities_x = col1.slider("Select the number of entities", min_value=1, max_value=20, value=10)
        
        data = {
            "Food": {
                "Quality": ["Taste", "Freshness", "Temperature", "Texture", "Appearance/Presentation", "Healthfulness", "Portion"],
                "Menu": ["Comments", "Inquiries"],
                "Issues": ["Availability", "Food Safety"]
            },
            "People": {
                "Team": ["Presentation", "Hospitality"]
            },
            "Experience": {
                "Payment": ["Cost of Meal", "Pricing Accuracy", "Payment Processing"],
                "Ordering": ["Speed of Service", "Order Accuracy", "Ordering Process"],
                "Loyalty": ["Loyalty"],
                "Amenities": ["Amenities"],
                "Inquiries": ["Inquiries"],
                "Cleanliness": ["Dining Room", "Kitchen", "Bathrooms", "Patio", "Drive-in", "Garbage"]
            }
        }

        category_options = list(data.keys())
        category = st.multiselect("Select categories", options=category_options, placeholder='All')
        if len(category) > 0:
            selected_category = category
        else:
            selected_category = category_options
        
        groups = []
        for cat in selected_category:
            groups.extend(data[cat].keys())
        group_options = list(set(groups))
        
        group = st.multiselect("Select groups", options=group_options, placeholder='All')
        if len(group) > 0:
            selected_group = group
            topics = []
            for g in selected_group:
                for cat in selected_category:
                    if g in data[cat]:
                        topics.extend(data[cat][g])
            topic_options = list(set(topics))
        else:
            selected_group = group_options
            topics = []
            for g in selected_group:
                for cat in selected_category:
                    if g in data[cat]:
                        topics.extend(data[cat][g])
            topic_options = list(set(topics))

        topic = st.multiselect("Select topics", options=topic_options, placeholder='All')        
        if len(topic) > 0:
            selected_topic = topic
        else:
            selected_topic = topic_options

        filtered_ids = sentences[sentences['CATEGORY'].isin(selected_category)]
        filtered_ids = filtered_ids[filtered_ids['CATEGORY_GROUP'].isin(selected_group)]
        filtered_ids = filtered_ids[filtered_ids['TOPIC'].isin(selected_topic)]
        filtered_ids = filtered_ids.drop_duplicates(subset=['FEEDBACK_ID'])
        filtered_review_data = filtered_review_data[filtered_review_data['FEEDBACK_ID'].isin(filtered_ids['FEEDBACK_ID'])]
        if not filtered_review_data.empty:
            filtered_entities = sentences[sentences['FEEDBACK_ID'].isin(filtered_review_data['FEEDBACK_ID'])]
        else:
            st.info("No reviews with feedback text available for the selected filters.", icon=':material/info:')
            st.stop()

    with col2:
        positive_entities = filtered_entities[filtered_entities['OVERALL_SENTIMENT'] == 'Positive']['ENTITIES'].value_counts().head(entities_x).sort_values(ascending=True)
        if not positive_entities.empty:
            fig_positive = px.bar(positive_entities, x=positive_entities.values, y=positive_entities.index, orientation='h', title='Positive Entities', text=positive_entities.values)
            fig_positive.update_layout(xaxis_title=None, yaxis_title=None)
            fig_positive.update_traces(marker_color='#34A853', textposition='inside')
            st.plotly_chart(fig_positive)
        else:
            st.success("No positive entities found for the selected filters.", icon=':material/info:')
    
    with col3:
        negative_entities = filtered_entities[filtered_entities['OVERALL_SENTIMENT'] == 'Negative']['ENTITIES'].value_counts().head(entities_x).sort_values(ascending=True)
        if not negative_entities.empty:
            fig_negative = px.bar(negative_entities, x=negative_entities.values, y=negative_entities.index, orientation='h', title='Negative Entities', text=negative_entities.values)
            fig_negative.update_layout(xaxis_title=None, yaxis_title=None)
            fig_negative.update_traces(marker_color='#FF5733', textfont_color='white', textposition='inside')  # Reddish-orange color
            st.plotly_chart(fig_negative)
        else:
            st.error("No negative entities found for the selected filters.", icon=':material/info:')

    #if not filtered_entities.empty and not pd.isna(filtered_entities['ENTITIES']).all():
     #   entities_text = " ".join(filtered_entities[filtered_entities['ENTITIES'].notna()]['ENTITIES'].astype(str).values)
      #  wordcloud = WordCloud(width=2500, height=250, background_color='white', colormap='CMRmap_r').generate(entities_text)
        
        # Create new figure and axis
       # fig, ax = plt.subplots(figsize=(10, 10))
        #ax.imshow(wordcloud, interpolation='bilinear')
        #ax.axis("off")
        
       # st.pyplot(fig)
    
    st.markdown("##### Review Details")
    st.dataframe(filtered_review_data[['FEEDBACK_DATE', 'FEEDBACK_TEXT','CATEGORY', 'CATEGORY_GROUP', 'TOPIC', 'ENTITIES', 'BRAND',
                                                'ADDRESS', 'AUTHOR', 'RATING', 'OVERALL_SENTIMENT', 'REVIEW_URL']].style.map(sentiment_color, subset=["OVERALL_SENTIMENT"]),
                column_config={
                    'FEEDBACK_DATE': 'Date',
                    'FEEDBACK_TEXT': st.column_config.Column(
                        'Feedback',
                        width="large"),
                    'CATEGORY': st.column_config.Column(
                        'Category',
                        width="medium"),
                    'CATEGORY_GROUP': st.column_config.Column(
                        'Group',
                        width="medium"),
                    'TOPIC': st.column_config.Column(
                        'Topic',
                        width="medium"),
                    'ENTITIES': st.column_config.Column(
                        'Entities',
                        width="medium"),
                    'BRAND': 'Brand',
                    'ADDRESS': st.column_config.Column(
                        'Location',
                        width="small"),
                    'AUTHOR': 'Author',
                    'RATING': 'Rating',
                    'OVERALL_SENTIMENT': 'Sentiment',
                    'REVIEW_URL': st.column_config.LinkColumn(
                        'Review URL',
                        width='small',
                        help='Link to the review',
                        display_text='URL')
                    },
                column_order=('FEEDBACK_DATE', 'RATING', 'FEEDBACK_TEXT', 'OVERALL_SENTIMENT', 'BRAND', 'ADDRESS', 'CATEGORY', 'CATEGORY_GROUP', 'TOPIC', 'ENTITIES', 'AUTHOR', 'REVIEW_URL'),
        hide_index=True, 
        use_container_width=True)
    

if menu_id == 'Support':
    st.markdown("<br>", unsafe_allow_html=True)
    filtered_review_data_detailed = filtered_review_data[filtered_review_data['FEEDBACK_TEXT'].notna()].sort_values('FEEDBACK_DATE', ascending=False)
    if filtered_review_data_detailed.empty:
        st.info('No reviews with feedback text available for the selected filters.', icon=':material/info:')
        st.stop()
    filtered_review_data_detailed['SELECT'] = [True] + [False] * (len(filtered_review_data_detailed) - 1)
    filtered_review_data_detailed['RATING'] = filtered_review_data_detailed['RATING'].astype(int)
    filtered_review_data_detailed['CUSTOMER_SUCCESS_NOTES'] = filtered_review_data_detailed['CUSTOMER_SUCCESS_NOTES'].fillna('')
    df_to_edit = st.data_editor(filtered_review_data_detailed[['SELECT', 'REVIEW_ID','AUTHOR', 'OVERALL_SENTIMENT', 'FEEDBACK_TEXT', 'RATING', 'ADDRESS',
                                                        'FEEDBACK_DATE', 'CUSTOMER_SUCCESS_NOTES', 'REVIEW_URL', 'STATUS', 'RESPONSE']].style.map(sentiment_color, subset=["OVERALL_SENTIMENT"]),
        column_order=('SELECT', 'FEEDBACK_DATE', 'AUTHOR', 'RATING', 'FEEDBACK_TEXT', 'OVERALL_SENTIMENT', 'STATUS', 'ADDRESS', 'REVIEW_URL', 'RESPONSE', 'CUSTOMER_SUCCESS_NOTES'), 
                column_config={
                    'SELECT': 'Select',
                    'FEEDBACK_DATE': 'Date',
                    'AUTHOR': st.column_config.Column(
                        "Author",
                        width="small"),
                    'OVERALL_SENTIMENT': 'Sentiment',
                    'FEEDBACK_TEXT': st.column_config.Column(
                        'Feedback',
                        width="large"),
                    'RATING': st.column_config.Column(
                        "Rating",
                        width="small"),
                    'CUSTOMER_SUCCESS_NOTES': 'Customer Success Notes',
                    'STATUS': st.column_config.SelectboxColumn(
                        'Status',
                        help="The status of the review",
                        width="small",
                        options=[
                            '🌱 New',
                            '✔️ Resolved',
                            '🚫 Spam',
                        ],
                    ),
                    'ADDRESS': st.column_config.Column(
                        "Location",
                        width="medium"),                            
                    'REVIEW_URL': st.column_config.LinkColumn(
                         'Review URL',
                         width='small',
                         help='Link to the review',
                         display_text='URL'),
                    'RESPONSE': 'Response'
                    },
                disabled=['OVERALL_SENTIMENT', 'FEEDBACK_TEXT', 'RATING', #'place_name', 
                            'FEEDBACK_DATE', 'AUTHOR', 'ADDRESS', 'REVIEW_URL', 'RESPONSE'],
                use_container_width=True, hide_index=True)
    
    if 'generated_responses' not in st.session_state:
        st.session_state['generated_responses'] = {}

    selected_sum = df_to_edit['SELECT'].sum()

    if selected_sum == 1:
        selected_review = df_to_edit.loc[df_to_edit['SELECT'] == True].iloc[0]
        review_text = selected_review['FEEDBACK_TEXT']
        author_name = selected_review['AUTHOR']
        prompt = f"""
Below is a restaurant review. Pretend you're the restaurant's social media manager and craft a concise (max 5 sentences), professional response. Where appropriate, acknowledge specific details from the review to personalize your reply. Start with a greeting, focus on addressing customer's feedback, and offering any necessary follow-up. Don't include any other text or comments. Return only the response.

Review:
{review_text}

Author: {author_name}
"""
        col8, col9 = st.columns(2, gap='medium', vertical_alignment='top')
        with col8:
            st.write(f'**Selected Review**')
            with st.container(border=True, height=170):
                col1, col2 = st.columns([0.05, 0.95], gap='medium')
                col1.write('🧑🏻')
                col2.write(f'{review_text}')
            col1, col2 = st.columns(2)
            placeholder = col2.empty()

        if placeholder.button('💬 Generate Response', use_container_width=True):
            if review_text in st.session_state['generated_responses']:
                response = st.session_state['generated_responses'][review_text]
            else:
                with st.spinner(':robot_face: Generating response, please wait...'):
                    response = generate_response(prompt)
                if response:
                    st.session_state['generated_responses'][review_text] = response
                else:
                    response = ''

        if review_text in st.session_state['generated_responses']:
            with col9:
                st.write(f'**Response Draft**')
                edited_response = st.text_area("Response Draft", st.session_state['generated_responses'][review_text], label_visibility='collapsed', height=170)
                col1, col2, col3 = st.columns(3)
                
                if col3.button('🔄 Regenerate', use_container_width=True):
                    st.session_state.regenerate_clicked = True

                if st.session_state.regenerate_clicked:
                    instruction = st.text_input("Additional instructions:", 
                                              key="regen_instruction",
                                              value=st.session_state.instruction)
                    
                    if instruction:
                        st.session_state.instruction = instruction
                        with st.spinner(':robot_face: Regenerating response, please wait...'):
                            new_prompt = f"""
Original task:
{prompt}

Previous response: {st.session_state['generated_responses'][review_text]}

Additional instruction: {instruction}

Please provide an updated response incorporating the additional instruction.
"""
                            response = generate_response(new_prompt)
                            if response:
                                st.session_state['generated_responses'][review_text] = response
                                st.session_state.regenerate_clicked = False  # Reset the clicked state
                                st.session_state.instruction = ''  # Reset the instruction
                                st.rerun()  # Rerun to show the new response
        
                if col3.button('💾 Save response', use_container_width=True):
                    # Get the review ID of the selected review
                    review_id = selected_review['REVIEW_ID']
                    
                    # Update the response in the filtered data
                    filtered_review_data_detailed['RESPONSE'] = filtered_review_data_detailed['RESPONSE'].astype('object')
                    filtered_review_data_detailed.loc[filtered_review_data_detailed['REVIEW_ID'] == review_id, 'RESPONSE'] = edited_response
                    
                    try:
                        # Create a DataFrame with just the updated row
                        update_df = pd.DataFrame({
                            'REVIEW_ID': [review_id],
                            'RESPONSE': [edited_response],
                            'STATUS': ['✔️ Resolved'] if selected_review['STATUS'] == '🌱 New' else [selected_review['STATUS']],
                            'CUSTOMER_SUCCESS_NOTES': [selected_review['CUSTOMER_SUCCESS_NOTES']]
                        })
                        # Ensure the RESPONSE column is string type
                        update_df['RESPONSE'] = update_df['RESPONSE'].astype(str)
                        update_df['STATUS'] = update_df['STATUS'].astype(str)
                        update_df['CUSTOMER_SUCCESS_NOTES'] = update_df['CUSTOMER_SUCCESS_NOTES'].astype(str)
                        
                        # Find the row with matching review ID and update the response
                        data.loc[data['REVIEW_ID'] == review_id, ['RESPONSE', 'STATUS', 'CUSTOMER_SUCCESS_NOTES']] = [
                            update_df['RESPONSE'].iloc[0],
                            update_df['STATUS'].iloc[0], 
                            update_df['CUSTOMER_SUCCESS_NOTES'].iloc[0]
                        ]
                        
                        # Create update dataframe with all columns from the original data
                        update_df = data[data['REVIEW_ID'] == review_id].copy()
                        
                        # Write to Keboola
                        write_table('out.c-257-data-model.MERGED_DATA_FINAL', update_df, is_incremental=True)
                        
                        st.success('Response saved successfully!')
                        
                    except Exception as e:
                        st.error(f'Failed to save response: {str(e)}')
                        
    elif selected_sum > 1:
        st.info('Select only one review to generate a response.')
    else:
        st.info('Select the review you want to respond to in the table above.')

if menu_id == 'Assistant':
    if st.session_state.thread_id is None:
        
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "To help you navigate the CSV file, here is the description of some important columns: "
                        "feedback_id: Unique identifier for the feedback. "
                        "feedback_channel: Source channel of the feedback (e.g., platform or app). "
                        "place_id: Unique identifier for the place being reviewed. "
                        "address: Address of the place. "
                        "place_total_score: The total score of the place based on reviews. "
                        "place_rev_count: Number of reviews for the place. "
                        "latitude/longitude: Geographical coordinates of the place. "   
                        "customer_name: Name of the customer. "
                        "reviewer_rev_count: Number of reviews by the reviewer. "
                        "feedback_date: Date when the feedback was given. " 
                        "rating: The rating given by the reviewer (on a scale). "
                        "review_content_meal_type: Type of meal mentioned in the review. "
                        "review_content_service_type: Type of service mentioned. "
                        "review_content_food/service/atmosphere: Specific ratings for food, service, and atmosphere. "  
                        "review_text: Text content of the review. "
                        "sentiment: Sentiment analysis result for the feedback (e.g., positive, negative). "
                        "city/state/postalCode: Location details of the place."
                    ),
                    "attachments": [
                        {
                        "file_id": FILE_ID, #file.id,
                        "tools": [{"type": "code_interpreter"}]
                        }
                    ]
                }
            ]
        )
        st.session_state.thread_id = thread.id

    if not st.session_state.table_written:
        df_log = pd.DataFrame({
            'thread_id': [st.session_state.thread_id],
            'created_at': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        })

        write_table(table_id='in.c-257-bot-log.logging', df=df_log, is_incremental=True)
        st.session_state.table_written = True

    with st.expander("Data"):
        data_to_display = data.copy()
        data_to_display.drop(columns=['REVIEW_ID', 'ID', 'FEEDBACK_ID', 'PLACE_ID', 'STREET', 'CITY', 'STATE', 'POSTAL_CODE', 'REVIEWER_NAME', 'CUSTOMER_ID', 'REVIEWER_ID', 'REVIEWER_URL', 'REVIEWER_NUM_OF_REVIEWS', 'REVIEW_DATE', 'EVENT_DATE', 'FEEDBACK_ORIGINAL_TEXT', 'REVIEW_TEXT'], errors='ignore', inplace=True)
        st.dataframe(data_to_display)
        st.caption(f'Thread ID: {st.session_state.thread_id}')


    for message in st.session_state.messages:
        if message["role"] == "user":
            avatar = '🧑‍💻'
        else:
            avatar = MINI_LOGO_URL

        with st.chat_message(message["role"], avatar=avatar):
            if "[Image:" in message["content"]:
                start_index = message["content"].find("[Image:") + len("[Image: ")
                end_index = message["content"].find("]", start_index)
                image_path = message["content"][start_index:end_index]
                st.image(image_path)
                
                text_content = message["content"][:start_index - len("[Image: ")] + message["content"][end_index + 1:]
                st.markdown(text_content)
            else:
                st.markdown(message["content"])
        placeholder = st.empty()
    styl = f"""
        <style>
            .stTextInput {{
                position: fixed;
                bottom: 1rem; /* Stick to the bottom of the viewport */
                background-color: white; /* Set background color to white */
                z-index: 1000; /* Bring the text input to the front */
                padding: 10px; /* Add some padding for aesthetics */
            }}
            .spacer {{
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                height: 2rem; /* Height of the spacer */
                background-color: #ffffff; /* Color of the spacer */
                z-index: 999; /* Ensure it's behind the text input */
            }}
        </style>
        """
    st.markdown(styl, unsafe_allow_html=True)
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)  # Add the spacer div
    
    text_input = st.text_input("Please enter your query:")
    if prompt := text_input:
        with placeholder.container():
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar='🧑‍💻'):
                st.markdown(prompt)

            with st.spinner('Analyzing...'):  
                thread_message = client.beta.threads.messages.create(
                    st.session_state.thread_id,
                    role="user",
                    content=prompt,
                )
                run = client.beta.threads.runs.create_and_poll(
                    thread_id=st.session_state.thread_id,
                    assistant_id=ASSISTANT_ID,
                )

            if run.status == 'completed':
                messages = client.beta.threads.messages.list(
                    thread_id=st.session_state.thread_id
                )
                newest_message = messages.data[0]
                complete_message_content = ""
                with st.chat_message("assistant", avatar=MINI_LOGO_URL):
                    for message_content in newest_message.content:
                        if hasattr(message_content, "image_file"):
                            file_id = message_content.image_file.file_id

                            resp = client.files.with_raw_response.retrieve_content(file_id)

                            if resp.status_code == 200:
                                image_data = BytesIO(resp.content)
                                img = Image.open(image_data)
                                
                                temp_dir = gettempdir()
                                image_path = os.path.join(temp_dir, f"{file_id}.png")
                                img.save(image_path)
                        
                                st.image(img)
                                complete_message_content += f"[Image: {image_path}]\n"

                        elif hasattr(message_content, "text"):
                            text = message_content.text.value
                            st.markdown(text)
                            complete_message_content += text + "\n"

                st.session_state.messages.append({"role": "assistant", "content": complete_message_content})

            else:
                st.write(f"Run status: {run.status}")
