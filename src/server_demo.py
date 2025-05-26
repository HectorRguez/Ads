import requests
import json
import time
from textwrap import fill
import difflib
from difflib import SequenceMatcher

# Server configuration
BASE_URL = 'http://localhost:8888'
HEADERS = {'Content-Type': 'application/json'}

def print_boxed_text(title, content, highlight_markers=None, width=78):
    """Helper function to print text in a box with optional highlighting."""
    border = "+" + "-" * (width) + "+"
    title_line = f"|{title.center(width)}|"
    
    lines = [border, title_line, border]
    
    # Wrap and process content
    wrapped_lines = fill(content, width=width-4).split('\n')
    
    for line in wrapped_lines:
        if highlight_markers:
            # Apply highlighting based on markers
            for start, end in highlight_markers.get(line, []):
                line = line[:start] + "**" + line[start:end] + "**" + line[end:]
        
        padded_line = f"| {line.ljust(width-2)} |"
        lines.append(padded_line)
    
    lines.append(border)
    return '\n'.join(lines)

def test_endpoint(name, method, endpoint, data=None):
    """Helper function to test endpoints with error handling"""
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"Endpoint: {method} {endpoint}")
    print(f"{'='*80}")
    
    try:
        response = requests.post(f"{BASE_URL}{endpoint}", headers=HEADERS, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Status Code: {response.status_code}")
            
            # Handle Q&A results (basic inference)
            if 'inferred' in result:
                question = result.get('question', 'N/A')
                answer = result.get('inferred', 'N/A')
                
                print("\nQ&A Generation Results:")
                print(print_boxed_text("QUESTION", question))
                print()
                print(print_boxed_text("ANSWER", answer))
            
            # Handle Q&A with native ads (combined inference)
            elif 'answer_with_ads' in result:
                question = result.get('question', 'N/A')
                original_answer = result.get('answer', 'N/A')
                answer_with_ads = result.get('answer_with_ads', 'N/A')
                
                print("\nQ&A with Native Ads Results:")
                print(print_boxed_text("QUESTION", question))
                print()
                print(print_boxed_text("ORIGINAL ANSWER", original_answer))
                print()
                print(print_boxed_text("ANSWER WITH ADS", answer_with_ads))
                
                if result.get('related_products'):
                    print(f"\nRelated Products Used:")
                    for prod in result['related_products'][:3]:
                        print(f"   - {prod['name']} (similarity: {prod['similarity']:.3f})")
            
            # Handle text with native ads (insert_native_ads)
            elif 'text_with_ads' in result:
                original_text = data.get('text', 'Original text not available') if data else 'Original text not available'
                text_with_ads = result['text_with_ads']
                
                print(print_boxed_text("ORIGINAL TEXT", original_text))
                print(print_boxed_text("TEXT WITH ADS", text_with_ads))

                if result.get('related_products'):
                    print(f"\nAll Related Products:")
                    for prod in result['related_products'][:3]:
                        print(f"   - {prod['name']} (similarity: {prod['similarity']:.3f})")
            
            return result
        else:
            print(f"❌ Error: {response.status_code}")
            error_text = response.text
            print(f"Response: {error_text}")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def test_health_check():
    """Test the health endpoint"""
    print(f"\n{'='*80}")
    print("Testing: Health Check")
    print(f"{'='*80}")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            result = response.json()
            print("✅ Server is healthy!")
            print(f"Status: {result['status']}")
            print("Model Status:")
            for model, status in result['models'].items():
                status_icon = "✅" if status else "❌"
                print(f"  {status_icon} {model}: {status}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check exception: {e}")

def main():
    print("Starting Flask RAG Server Demo with Native Advertising")
    
    # Health check first
    test_health_check()
    
    # Test native ad insertion with existing text
    print(f"\n{'#'*80}")
    print("NATIVE AD INSERTION TESTS (existing text)")
    print(f"{'#'*80}")
    
    ad_test_cases = [
        ("""What is in a .ckpt file for machine learning?""",
         """
        A .ckpt file, short for "checkpoint" file, is used in machine learning to store the current state of a trained model. It typically contains the following information:

1. Model architecture: The structure of the neural network, such as the number and types of layers, along with their configurations.

2. Model weights: The learned parameters or weights of the model at a specific point during training.

3. Optimizer state: The state of the optimizer, including details like learning rate, momentum, and other hyperparameters, at the time of saving the checkpoint.

4. Training configuration: The training settings, such as the batch size, loss function, and number of epochs.

Checkpoint files are essential for resuming training from a specific point, allowing for more efficient retraining and fine-tuning of models, and they are also useful for saving and sharing the best performing models during the training process. The main goal of checkpointing is to enable recovery from interruptions, prevent loss of progress, and facilitate model evaluation at different stages during training.
        """),
        ("""How can I start an ethereum node?""",
         """
        There are different ways to start an Ethereum node, but here are the basic steps:

1. Download and install an Ethereum client. You can choose from different clients like Geth, Parity, and Besu. Check the documentation and system requirements for each client before downloading.

2. Sync the client with the Ethereum network. This process may take several hours or days, depending on the client and your internet connection. You can sync using the command line interface or a graphical user interface (GUI).

3. Configure the client. You can customize various settings like the gas price, block time, and network ID. Consult the documentation for the specific client you're using for more details.

4. Get some testnet or mainnet ether to use for testing or running applications on the network.

5. Run the client as a node. This allows you to connect to the Ethereum network and participate in transactions, mining, or other activities. You can check the node status and data using the client's console or web interface.

6. Monitor and update your node regularly. Keep the client updated with the latest version and security patches to ensure smooth operation and protection against attacks.
        """),
        ("""where can I download free audiobook""",
         """
        There are several websites where you can download free audiobooks, such as:

1. Librivox (https://librivox.org/) - offers public domain audiobooks read by volunteers from all over the world.
2. Loyal Books (http://www.loyalbooks.com/) - offers a wide range of public domain audiobooks.
3. Open Culture (http://www.openculture.com/freeaudiobooks) - a curated list of free audiobooks.
4. Lit2Go (https://etc.usf.edu/lit2go/) - a free educational audiobook website by the University of South Florida.
5. Project Gutenberg (http://www.gutenberg.org/) - offers a limited selection of audiobooks, but has a large collection of eBooks in text format.
6. Storynory (https://www.storynory.com/) - children's audiobooks and fairy tales.
7. Audiobook Treasury (https://www.audiobooktreasury.com/free-audiobooks/) - offers a small selection of free audiobooks.

Please note that most of these websites focus on public domain (out-of-copyright) books, so you might not find the latest bestsellers there.
        """),
        ("""How to earn passive income with my web development skills in 2023?""",
        """
        As an AI language model, I do not have preferences, but here are some ways to earn passive income with your web development skills in 2023:

1. Building and selling web applications: You can develop a web application or a software tool that can automate a common problem or streamline a work process. You can sell these applications on platforms like AppSumo, Gumroad or SmartCents.

2. Create WordPress Themes and Plugins: WordPress powers a significant amount of the internet, which means businesses and website owners are always interested in ways to customize their website. You can create and sell WordPress Themes or Plugins on ThemeForest or CodeCanyon.

3. Affiliate marketing: If you have experience in web development, you can create web development tutorials on popular platforms like Udemy, Skillshare, and YouTube. You can share your affiliate links to useful tools like SiteGround Hosting, Elementor, GeneratePress, or ReactJS.

4. Create and sell stock photos and videos: As a web developer, you must have a keen eye for design. You can take advantage of this skill by creating stock photos and videos of web-related images like coding on a laptop or designing graphic elements for a website. You can sell these on platforms like DepositPhotos and iStock.

5. Buy and sell websites: You can also make a good profit by buying and reselling established websites that generate passive income. You can acquire websites on platforms like Flippa and  FE International, and then optimizing them to increase their value before reselling them to a higher price."""
        ),
        ("""here simple javascript to replace words in html document using table""",
            """
            You can write a simple JavaScript script like this to replace words in an HTML document using a table (in this case, an object that maps the words to their replacements).

1. Include the following script inside a `<script>` tag in your HTML file:

```javascript
document.addEventListener("DOMContentLoaded", function() {
  const replacements = {
    "word1": "replacement1",
    "word2": "replacement2",
    // Add more words and replacements here
  };

  function replaceWords(node) {
    if (node.nodeType === Node.TEXT_NODE) {
      Object.keys(replacements).forEach((word) => {
        const regex = new RegExp('\\b' + word + '\\b', 'g');
        node.textContent = node.textContent.replace(regex, replacements[word]);
      });
    } else if (node.nodeType === Node.ELEMENT_NODE) {
      node.childNodes.forEach(replaceWords);
    }
  }

  replaceWords(document.body);
});
```

2. Replace "word1", "replacement1", "word2", "replacement2" in the `replacements` object with the words you want to replace and their respective replacements.

3. When you load the HTML document in the browser, the script replaces all occurrences of the specified words with their replacements.

Note: This script will only run if the page is loaded in a web browser. To test it, you can create an HTML file with some text containing the words you want to replace, and then run the file in the browser.
            """
        ),
        ("""What is the future of the Prolog programming language?""",
         """As an AI language model, I cannot predict the exact future of the Prolog programming language, but based on current trends and developments, it is likely that Prolog will continue to be used for certain specialized applications, such as knowledge representation and automated reasoning. However, other languages such as Python, Java, and MATLAB are more popular in many areas of computer science and engineering, and may continue to dominate the mainstream. Additionally, new developments such as logic-based AI and machine learning may create new opportunities for Prolog to be used in innovative ways. Overall, the future of Prolog will likely depend on its ability to adapt to changing technology and meet the demands of different industries and applications.
         """
        )
    ]
    
    for i, (prompt, test_case) in enumerate(ad_test_cases, 1):
        test_endpoint(
            f"Text Ad Insertion Test {i}", 
            "POST", 
            "/insert_native_ads",
            {
                'prompt': prompt,
                'text': test_case,
                'max_tokens': 4096
            }
        )
        time.sleep(1)
    
    # Main feature: Combined Q&A with Native Ads (most representative)
    print(f"\n{'#'*80}")
    print("MAIN FEATURE: Q&A WITH NATIVE ADS")
    print(f"{'#'*80}")
    
    # Test questions for combined inference with ads
    test_questions_with_ads = [
        {
            'question': 'What is cybersecurity and why is it important?',
            'max_tokens': 500
        },
        {
            'question': 'How can businesses protect themselves from cyber threats?',
            'max_tokens': 500
        }
    ]
    
    # Test both local and remote inference with ads for each question
    for i, test_data in enumerate(test_questions_with_ads, 1):        
        # Test local inference with ads
        local_result = test_endpoint(
            f"Local Q&A with Native Ads {i}", 
            "POST", 
            "/infer_local_native_ads",
            test_data
        )
        
        time.sleep(2)
        
        # Test remote inference with ads
        remote_result = test_endpoint(
            f"Remote Q&A with Native Ads {i}", 
            "POST", 
            "/infer_remote_native_ads",
            test_data
        )
        
        time.sleep(2)
    
if __name__ == "__main__":
    main()
