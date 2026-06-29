# Upload to the existing GitHub repository

Extract the ZIP, then copy its contents into your local clone:

```bash
git clone https://github.com/BonicBruh/Pokemon-TCG-AI.git
cd Pokemon-TCG-AI
# Copy the extracted files into this folder, then:
git add .
git commit -m "Add CABT PPO training pipeline"
git push origin main
```

Do not commit downloaded daily replay datasets or large model checkpoints unless you intentionally use Git LFS. The included `.gitignore` excludes them.
