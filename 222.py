import streamlit as st
import pandas as pd
from fastai.learner import load_learner

# 加载模型
model = load_learner('model6.pkl')

# 加载笑话数据
jokes_df = pd.read_excel('Dataset4JokeSet.xlsx')
jokes_df.columns = ['joke']
jokes_df = jokes_df.rename_axis('joke_id').reset_index()


# 假设的推荐函数，根据用户评分推荐笑话
def recommend_jokes(user_ratings, n=5):
    user_id = max(user_ratings['user_id']) + 1  
    new_data = pd.DataFrame({'user_id': [user_id]*len(jokes_df), 'joke_id': jokes_df['joke_id']})
    dls = model.dls.test_dl(new_data)  
    preds, _ = model.get_preds(dl=dls)
    new_data['rating'] = preds
    new_data = new_data.sort_values(by='rating', ascending=False).head(n)
    return new_data.merge(jokes_df, on='joke_id')

# 初始化会话状态
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = pd.DataFrame(columns=['user_id', 'joke_id', 'rating'])

# Streamlit应用程序
st.title('笑话推荐系统')

# 初始界面随机出现3个笑话
if 'initial_jokes' not in st.session_state:
    st.session_state.initial_jokes = jokes_df.sample(3)

# 让用户对随机生成的笑话进行评分
st.write("初始界面随机出现3个笑话，请分别进行评分：")
for index, row in st.session_state.initial_jokes.iterrows():
    joke_id = row['joke_id']
    joke_text = row['joke']
    st.write(f"笑话 {joke_id + 1}: {joke_text}")
    score = st.slider(f"笑话 {joke_id + 1} 的评分:", min_value=0, max_value=5, value=3, key=f"initial_joke_{joke_id}")
    # 使用loc方法添加行到会话状态中的DataFrame
    st.session_state.user_ratings.loc[len(st.session_state.user_ratings)] = [0, joke_id, score]

# 提交评分按钮
submit_scores = st.button("提交评分")

if submit_scores:
    # 根据用户的评分，推荐5个新的笑话
    recommended_jokes = recommend_jokes(st.session_state.user_ratings, n=5)
    st.write("\n根据你的评分，推荐以下5个笑话：")
    st.session_state.user_ratings_recommended = pd.DataFrame(columns=['user_id', 'joke_id', 'rating'])

    for index, row in recommended_jokes.iterrows():
        joke_id = row['joke_id']
        joke_text = row['joke']
        st.write(f"笑话 {joke_id + 1}: {joke_text}")
        score = st.slider(f"笑话 {joke_id + 1} 的评分:", min_value=0, max_value=5, value=3, key=f"recommended_joke_{joke_id}")
        # 使用loc方法添加行到会话状态中的DataFrame
        st.session_state.user_ratings_recommended.loc[len(st.session_state.user_ratings_recommended)] = [0, joke_id, score]

    # 计算用户满意度
    satisfaction_score = st.session_state.user_ratings_recommended['rating'].mean()
    st.write(f"\n本次推荐的用户满意度为：{satisfaction_score:.2f}/5")
