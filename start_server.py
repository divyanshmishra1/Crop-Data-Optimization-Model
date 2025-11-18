import subprocess
from pyngrok import ngrok

port = 8501
ngrok.set_auth_token("YOUR_NGROK_TOKEN")
public_url = ngrok.connect(port)
print("PUBLIC URL:", public_url)

subprocess.run(["streamlit", "run", "app.py"])
