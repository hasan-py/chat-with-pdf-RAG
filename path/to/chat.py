<<<<<<< SEARCH
        if related_documents:
            answer = answer_question(question, related_documents)
            html_answer = "<div style='display: inline-block; padding: 10px; border: 1px solid black;'>"
            for tag in answer.split("<"):
                 if tag.startswith("</") or not tag.startswith("<"):
                     continue
                 html_answer += f"<span style='font-weight: bold'>{tag[1:-1]}</span>"
            html_answer += "</div>"
            st.chat_message("assistant").markdown(f"{html_answer}")
=======
        if related_documents:
            answer = answer_question(question, related_documents)
            
            # Process the answer to remove standalone HTML tags
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(answer, 'html.parser')
            processed_answer = str(soup.get_text())
            
            html_answer = f"<div style='display: inline-block; padding: 10px; border: 1px solid black; margin-bottom: 20px;'>{processed_answer}</div>"
            st.chat_message("assistant").markdown(html_answer)
>>>>>>> REPLACE
