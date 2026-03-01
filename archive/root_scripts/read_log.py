
try:
    with open('test_output.txt', 'r', encoding='utf-16') as f:
        content = f.read()
except:
    with open('test_output.txt', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

print("--- START OF LOG ---")
print(content)
print("--- END OF LOG ---")
