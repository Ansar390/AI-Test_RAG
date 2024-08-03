



"""**SOLUTION 2**"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import api_keys

llm_grok = ChatGroq(

            groq_api_key=api_keys.os.environ["GROQ_API_KEY"],

            model_name='llama-3.1-8b-instant', temperature=0.0

    )

template ="""You are intelligent chatbot designed for users to ask questions about a conversation between a doctor and a patient. You will answer the relevant information from the provided transcript and answer user queries.
You have the database below

Doctor (D): Good morning, how are you feeling today?
Patient (P): Good morning, Doctor. I've been feeling very anxious and stressed lately.
D: I'm sorry to hear that. Can you describe your symptoms in more detail?
P: I've been having trouble sleeping, my heart races for no reason, and I often feel
like I'm on edge. I also feel exhausted all the time.
D: It sounds like you might be experiencing symptoms of Generalized Anxiety Disorder (GAD). Have you experienced these symptoms before?
P: Yes, I've had anxiety for a few years, but it's gotten worse recently.
D: I understand. Based on your symptoms and history, I'm diagnosing you with Generalized Anxiety Disorder. We'll need to address this with a combination of medication, therapy, and lifestyle changes. Does that sound okay to you?
P: Yes, I just want to feel better.
D: For medication, I'm going to prescribe you an SSRI (Selective Serotonin Reuptake Inhibitor) called Sertraline. This should help manage your anxiety symptoms. It's important to take it as prescribed and be patient, as it may take a few weeks to see the full eects.
P: Okay, I can do that.
D: In addition to the medication, I'd like you to try some cognitive-behavioral therapy (CBT). This type of therapy can help you identify and change negative thought patterns and behaviors. I'll refer you to a therapist who specializes in CBT.
P: That sounds helpful. I've heard of CBT before.
D: Great. Now, let's talk about some exercises and lifestyle changes. Regular physical exercise can be very beneficial for reducing anxiety. Aim for at least 30 minutes of moderate exercise, like walking or yoga, most days of the week.
P: I can try to incorporate that into my routine.
D: Good. Also, practicing mindfulness or meditation daily can help reduce stress. There are
many apps and online resources that can guide you through these practices. P: I've never tried meditation, but I'm willing to give it a go.
D: Excellent. Finally, let's discuss some precautions. Avoid caeine and alcohol as they can worsen anxiety symptoms. Make sure to get enough sleep, and try to maintain a regular sleep schedule.
P: I do drink a lot of coee. I'll try to cut back.
D: It's all about making small, sustainable changes. We will monitor your progress closely and adjust the treatment plan as needed. Do you have any questions or concerns?
P: Not at the moment. Thank you, Doctor.
D: You're welcome. Remember, you're not alone in this, and we're here to support
you. I'll see you in two weeks for a follow-up. P: Thank you, Doctor. I appreciate it.




Question: {question}

"""

prompt = ChatPromptTemplate.from_template(template)

output_parser= StrOutputParser()
chain= ({
    "question": RunnablePassthrough()

}
| prompt
| llm_grok
| output_parser
)


while True:
    query=input("Enter your query: ")
    if query=="":
        break
    else:

        for chunk in chain.stream(query):
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
            else:
                print(chunk.content, end="", flush=True)

#- What doctor diagnosed?
#- What medicine doctor mentioned?
#- Duration of medicine?
#- Precautions if any?
#- Activity if any?
