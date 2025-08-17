# Deployment Guide for StudyMate.ai

This guide covers how to deploy StudyMate.ai to various hosting platforms.

## 🚀 Quick Deployment Options

### 1. Streamlit Cloud (Recommended - Free)

**Steps:**
1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your repository and the `online` folder
5. Set main file path: `ui.py`
6. Deploy!

**Pros:** Free, easy, automatic updates from GitHub
**Cons:** Limited resources, may sleep after inactivity

### 2. Heroku (Free Tier Available)

**Steps:**
1. Install Heroku CLI
2. Create a new Heroku app: `heroku create your-app-name`
3. Push to Heroku: `git push heroku main`
4. Open your app: `heroku open`

**Files needed:** ✅ Procfile included
**Pros:** Reliable, good for production
**Cons:** May have cold starts on free tier

### 3. Railway (Modern Alternative)

**Steps:**
1. Go to [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Select the `online` folder as root
4. Set start command: `streamlit run ui.py --server.port $PORT --server.address 0.0.0.0`
5. Deploy!

**Pros:** Fast, modern, good free tier
**Cons:** Newer platform

### 4. Render (Free Static Sites)

**Steps:**
1. Go to [render.com](https://render.com)
2. Connect your GitHub repository
3. Create a new Web Service
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `streamlit run ui.py --server.port $PORT --server.address 0.0.0.0`

**Pros:** Free tier, good performance
**Cons:** Cold starts on free tier

## 🔧 Configuration for Online Deployment

### Environment Variables (Optional)

If you want to use environment variables instead of hardcoded API keys:

1. **Streamlit Cloud:** Add in app settings
2. **Heroku:** Use `heroku config:set KEY=value`
3. **Railway/Render:** Add in dashboard settings

**Recommended environment variables:**
```
GOOGLE_API_KEY=your_gemini_api_key
GOOGLE_API_KEY_BACKUP=your_backup_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
```

### Memory Considerations

The app uses AI models that require memory:
- **Minimum:** 1GB RAM
- **Recommended:** 2GB+ RAM
- **For heavy usage:** 4GB+ RAM

**Platform memory limits:**
- Streamlit Cloud: ~1GB
- Heroku Free: 512MB (may need paid tier)
- Railway: 1GB free, upgradeable
- Render: 512MB free, upgradeable

## 📁 File Structure for Deployment

```
online/
├── ui.py                    # Main app
├── config.py               # Configuration
├── llm_generator.py        # AI generation
├── pdf_processor.py        # PDF processing
├── embeddings.py           # Document search
├── quiz_manager.py         # Quiz functionality
├── pdf_generator.py        # PDF export
├── requirements.txt        # Dependencies
├── Procfile               # Heroku config
├── setup.sh               # Streamlit setup
├── .streamlit/
│   └── config.toml        # Streamlit config
├── .gitignore             # Git ignore rules
├── README.md              # Documentation
└── DEPLOYMENT.md          # This file
```

## 🔒 Security Notes

1. **API Keys:** Consider using environment variables for production
2. **File Uploads:** The app processes user-uploaded PDFs
3. **Memory Usage:** Monitor memory usage with large PDFs
4. **Rate Limits:** Be aware of API rate limits for Gemini

## 🐛 Troubleshooting

### Common Issues:

1. **Memory Errors:**
   - Upgrade to paid tier with more RAM
   - Optimize model loading in code

2. **Slow Loading:**
   - Models download on first run
   - Consider model caching strategies

3. **API Errors:**
   - Check API key configuration
   - Verify API quotas and limits

4. **PDF Processing Errors:**
   - Ensure PyMuPDF is properly installed
   - Check file size limits on platform

### Platform-Specific Issues:

**Streamlit Cloud:**
- May timeout on large model downloads
- Limited to 1GB RAM

**Heroku:**
- Slug size limit (500MB)
- May need buildpacks for some dependencies

**Railway/Render:**
- Check build logs for dependency issues
- Ensure proper port configuration

## 📞 Support

If you encounter deployment issues:
1. Check the platform's documentation
2. Review error logs
3. Create an issue in the GitHub repository
4. Check community forums for the hosting platform

## 🎯 Recommended Deployment Path

For beginners: **Streamlit Cloud**
For production: **Railway** or **Render**
For enterprise: **Heroku** or custom hosting
