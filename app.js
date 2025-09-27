class RAGSystem {
    constructor() {
        this.documents = new Map();
        this.chunks = [];
        this.currentPdf = null;
        this.currentPage = 1;
        this.totalPages = 0;
        this.questionCount = 0;
        this.initializeElements();
        this.setupEventListeners();
    }

    initializeElements() {
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.status = document.getElementById('status');
        this.queryInput = document.getElementById('queryInput');
        this.askButton = document.getElementById('askButton');
        this.clearButton = document.getElementById('clearButton');
        this.answers = document.getElementById('answers');
        this.pdfViewer = document.getElementById('pdfViewer');
        this.prevPage = document.getElementById('prevPage');
        this.nextPage = document.getElementById('nextPage');
        this.pageInfo = document.getElementById('pageInfo');
    }

    setupEventListeners() {
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        this.askButton.addEventListener('click', this.handleQuestion.bind(this));
        this.clearButton.addEventListener('click', this.clearAll.bind(this));
        this.queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !this.askButton.disabled) {
                this.handleQuestion();
            }
        });
        this.prevPage.addEventListener('click', () => this.changePage(-1));
        this.nextPage.addEventListener('click', () => this.changePage(1));
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files).filter(f => f.type === 'application/pdf');
        if (files.length > 0) {
            this.processFiles(files);
        }
    }

    handleFileSelect(e) {
        const files = Array.from(e.target.files);
        if (files.length > 0) {
            this.processFiles(files);
        }
    }

    async processFiles(files) {
        this.showStatus('Processing PDF documents...', 'processing');
        
        try {
            for (const file of files) {
                await this.processPDF(file);
            }
            
            this.showStatus(`Successfully processed ${files.length} document(s)`, 'success');
            this.queryInput.disabled = false;
            this.askButton.disabled = false;
            
            // Load first document for viewing
            if (this.documents.size > 0) {
                const firstDoc = this.documents.values().next().value;
                await this.loadPDFViewer(firstDoc.file);
            }
        } catch (error) {
            this.showStatus('Error processing PDF: ' + error.message, 'error');
        }
    }

    async processPDF(file) {
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        
        const chunks = [];
        for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const textContent = await page.getTextContent();
            const text = textContent.items.map(item => item.str).join(' ');
            
            // Create chunks from the page text
            const pageChunks = this.createChunks(text, file.name, i);
            chunks.push(...pageChunks);
        }
        
        this.documents.set(file.name, {
            file: file,
            pdf: pdf,
            chunks: chunks
        });
        
        this.chunks.push(...chunks);
    }

    createChunks(text, filename, pageNum, chunkSize = 500) {
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
        const chunks = [];
        let currentChunk = '';
        
        for (const sentence of sentences) {
            if ((currentChunk + sentence).length > chunkSize && currentChunk.length > 0) {
                chunks.push({
                    text: currentChunk.trim(),
                    source: filename,
                    page: pageNum,
                    id: `${filename}_${pageNum}_${chunks.length}`
                });
                currentChunk = sentence;
            } else {
                currentChunk += (currentChunk ? '. ' : '') + sentence;
            }
        }
        
        if (currentChunk.trim()) {
            chunks.push({
                text: currentChunk.trim(),
                source: filename,
                page: pageNum,
                id: `${filename}_${pageNum}_${chunks.length}`
            });
        }
        
        return chunks;
    }

    async loadPDFViewer(file) {
        const arrayBuffer = await file.arrayBuffer();
        this.currentPdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        this.totalPages = this.currentPdf.numPages;
        this.currentPage = 1;
        
        this.updatePageControls();
        await this.renderPage();
    }

    async renderPage() {
        if (!this.currentPdf) return;
        
        const page = await this.currentPdf.getPage(this.currentPage);
        const viewport = page.getViewport({ scale: 1.2 });
        
        const canvas = document.createElement('canvas');
        canvas.className = 'pdf-page';
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        
        const context = canvas.getContext('2d');
        await page.render({ canvasContext: context, viewport: viewport }).promise;
        
        this.pdfViewer.innerHTML = '';
        this.pdfViewer.appendChild(canvas);
    }

    updatePageControls() {
        this.prevPage.disabled = this.currentPage <= 1;
        this.nextPage.disabled = this.currentPage >= this.totalPages;
        this.pageInfo.textContent = `Page ${this.currentPage} of ${this.totalPages}`;
    }

    changePage(direction) {
        const newPage = this.currentPage + direction;
        if (newPage >= 1 && newPage <= this.totalPages) {
            this.currentPage = newPage;
            this.updatePageControls();
            this.renderPage();
        }
    }

    async handleQuestion() {
        const query = this.queryInput.value.trim();
        if (!query || this.chunks.length === 0) return;

        this.askButton.disabled = true;
        this.askButton.innerHTML = '<div class="loading"></div> Processing...';

        try {
            const relevantChunks = this.retrieveRelevantChunks(query);
            const answer = this.generateAnswer(query, relevantChunks);
            
            this.displayAnswer(query, answer, relevantChunks);
            this.queryInput.value = '';
        } catch (error) {
            this.showStatus('Error processing question: ' + error.message, 'error');
        } finally {
            this.askButton.disabled = false;
            this.askButton.textContent = 'Ask Question';
        }
    }

    retrieveRelevantChunks(query, topK = 5) {
        const queryTerms = query.toLowerCase().split(/\s+/);
        
        const scoredChunks = this.chunks.map(chunk => {
            const chunkText = chunk.text.toLowerCase();
            let score = 0;
            
            // Simple keyword matching with TF-IDF-like scoring
            for (const term of queryTerms) {
                const termCount = (chunkText.match(new RegExp(term, 'g')) || []).length;
                if (termCount > 0) {
                    // Boost score for exact matches and multiple occurrences
                    score += termCount * (term.length > 3 ? 2 : 1);
                }
            }
            
            // Boost score for chunks that contain multiple query terms
            const matchingTerms = queryTerms.filter(term => chunkText.includes(term));
            score *= Math.pow(matchingTerms.length, 0.5);
            
            return { chunk, score };
        });

        return scoredChunks
            .filter(item => item.score > 0)
            .sort((a, b) => b.score - a.score)
            .slice(0, topK)
            .map(item => item.chunk);
    }

    generateAnswer(query, chunks) {
        if (chunks.length === 0) {
            return "I couldn't find relevant information in the uploaded documents to answer your question.";
        }

        // Simple extractive summarization
        const relevantSentences = [];
        const queryTerms = query.toLowerCase().split(/\s+/);
        
        for (const chunk of chunks) {
            const sentences = chunk.text.split(/[.!?]+/).filter(s => s.trim().length > 20);
            
            for (const sentence of sentences) {
                const sentenceLower = sentence.toLowerCase();
                const matchCount = queryTerms.filter(term => sentenceLower.includes(term)).length;
                
                if (matchCount > 0) {
                    relevantSentences.push({
                        text: sentence.trim(),
                        matches: matchCount,
                        source: chunk.source,
                        page: chunk.page
                    });
                }
            }
        }
        
        // Sort by relevance and take top sentences
        relevantSentences.sort((a, b) => b.matches - a.matches);
        const topSentences = relevantSentences.slice(0, 3);
        
        if (topSentences.length === 0) {
            return "I found some relevant content but couldn't extract specific information to answer your question.";
        }
        
        return "Based on the documents: " + topSentences.map(s => s.text).join('. ');
    }

    displayAnswer(question, answer, chunks) {
        this.questionCount++;
        
        const answerCard = document.createElement('div');
        answerCard.className = 'answer-card';
        
        const questionDiv = document.createElement('div');
        questionDiv.className = 'question';
        questionDiv.textContent = `Q${this.questionCount}: ${question}`;
        
        const answerDiv = document.createElement('div');
        answerDiv.className = 'answer';
        answerDiv.textContent = answer;
        
        const citationsDiv = document.createElement('div');
        citationsDiv.className = 'citations';
        citationsDiv.innerHTML = '<strong>Sources:</strong><br>';
        
        chunks.forEach((chunk, index) => {
            const citation = document.createElement('div');
            citation.className = 'citation';
            citation.textContent = `[${index + 1}] ${chunk.source} - Page ${chunk.page}`;
            citation.addEventListener('click', () => this.highlightEvidence(chunk));
            citationsDiv.appendChild(citation);
        });
        
        answerCard.appendChild(questionDiv);
        answerCard.appendChild(answerDiv);
        answerCard.appendChild(citationsDiv);
        
        this.answers.insertBefore(answerCard, this.answers.firstChild);
    }

    async highlightEvidence(chunk) {
        // Switch to the document containing the evidence
        const doc = this.documents.get(chunk.source);
        if (doc && doc !== this.currentPdf) {
            await this.loadPDFViewer(doc.file);
        }
        
        // Navigate to the correct page
        if (chunk.page !== this.currentPage) {
            this.currentPage = chunk.page;
            this.updatePageControls();
            await this.renderPage();
        }
        
        // Add visual highlight effect to the viewer
        this.pdfViewer.style.border = '3px solid #FFD700';
        this.pdfViewer.classList.add('highlight');
        
        setTimeout(() => {
            this.pdfViewer.style.border = '2px solid #e1e5e9';
            this.pdfViewer.classList.remove('highlight');
        }, 3000);
        
        // Scroll the evidence into view
        this.pdfViewer.scrollIntoView({ behavior: 'smooth' });
    }

    showStatus(message, type) {
        this.status.innerHTML = `<div class="status ${type}">${message}</div>`;
        
        if (type !== 'processing') {
            setTimeout(() => {
                this.status.innerHTML = '';
            }, 5000);
        }
    }

    clearAll() {
        this.documents.clear();
        this.chunks = [];
        this.currentPdf = null;
        this.currentPage = 1;
        this.totalPages = 0;
        this.questionCount = 0;
        
        this.queryInput.value = '';
        this.queryInput.disabled = true;
        this.askButton.disabled = true;
        this.answers.innerHTML = '';
        this.pdfViewer.innerHTML = '<div style="text-align: center; padding: 50px; color: #666;">Upload a PDF document to begin</div>';
        this.pageInfo.textContent = 'No document loaded';
        this.prevPage.disabled = true;
        this.nextPage.disabled = true;
        this.status.innerHTML = '';
        
        this.showStatus('All documents cleared', 'success');
    }
}

// Initialize the RAG system when the page loads
window.addEventListener('load', () => {
    // Set up PDF.js worker
    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
    
    new RAGSystem();
});