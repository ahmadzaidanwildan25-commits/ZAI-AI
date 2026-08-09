import 'package:flutter/material.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  final List<_ChatMessage> _messages = <_ChatMessage>[];

  bool _isTyping = false;
  int _requestId = 0;

  Future<void> _sendMessage() async {
    final String text = _controller.text.trim();

    if (text.isEmpty || _isTyping) {
      return;
    }

    final int currentRequestId = ++_requestId;

    setState(() {
      _messages.add(
        _ChatMessage(
          text: text,
          isUser: true,
        ),
      );

      _controller.clear();
      _isTyping = true;
    });

    _scrollToBottom();

    /*
     * TEMPORARY AI RESPONSE
     *
     * Ini masih simulasi.
     * Pada tahap berikutnya bagian ini akan diganti
     * dengan koneksi ke ZAI AI Core / API Service.
     */
    await Future<void>.delayed(
      const Duration(milliseconds: 900),
    );

    if (!mounted || currentRequestId != _requestId) {
      return;
    }

    setState(() {
      _isTyping = false;

      _messages.add(
        const _ChatMessage(
          text:
              'Perintah diterima. ZAI Core siap menerima instruksi. AI engine akan kita sambungkan pada tahap berikutnya.',
          isUser: false,
        ),
      );
    });

    _scrollToBottom();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted || !_scrollController.hasClients) {
        return;
      }

      final double maxScroll =
          _scrollController.position.maxScrollExtent;

      _scrollController.animateTo(
        maxScroll,
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeOut,
      );
    });
  }

  void _quickCommand(String command) {
    if (_isTyping) {
      return;
    }

    _controller
      ..text = command
      ..selection = TextSelection.collapsed(
        offset: command.length,
      );

    _sendMessage();
  }

  @override
  void dispose() {
    _requestId++;

    _controller.dispose();
    _scrollController.dispose();

    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF020911),
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(),
            Expanded(
              child: _messages.isEmpty
                  ? _buildWelcome()
                  : _buildMessages(),
            ),
            _buildQuickCommands(),
            _buildInput(),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: 28,
        vertical: 20,
      ),
      decoration: const BoxDecoration(
        color: Color(0xFF030B14),
        border: Border(
          bottom: BorderSide(
            color: Color(0xFF123044),
          ),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
              color: const Color(0xFF062432),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(
                color: const Color(0xFF00D9FF),
              ),
            ),
            child: const Icon(
              Icons.chat_bubble_outline,
              color: Color(0xFF00D9FF),
              size: 25,
            ),
          ),
          const SizedBox(width: 16),
          const Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'ZAI CHAT',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 2,
                  ),
                ),
                SizedBox(height: 4),
                Text(
                  'INTELLIGENCE CONVERSATION INTERFACE',
                  style: TextStyle(
                    color: Color(0xFF7190A5),
                    fontSize: 11,
                    letterSpacing: 1.5,
                  ),
                ),
              ],
            ),
          ),
          _onlineIndicator(),
        ],
      ),
    );
  }

  Widget _onlineIndicator() {
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: 14,
        vertical: 8,
      ),
      decoration: BoxDecoration(
        color: const Color(0xFF03231E),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: const Color(0xFF00D9A5),
        ),
      ),
      child: const Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.circle,
            color: Color(0xFF00E6A8),
            size: 9,
          ),
          SizedBox(width: 8),
          Text(
            'ONLINE',
            style: TextStyle(
              color: Color(0xFF00E6A8),
              fontSize: 11,
              fontWeight: FontWeight.bold,
              letterSpacing: 1,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildWelcome() {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(30),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: 130,
              height: 130,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: const Color(0xFF031A27),
                border: Border.all(
                  color: const Color(0xFF00D9FF),
                  width: 2,
                ),
                boxShadow: const [
                  BoxShadow(
                    color: Color(0x4400D9FF),
                    blurRadius: 35,
                    spreadRadius: 5,
                  ),
                ],
              ),
              child: const Center(
                child: Text(
                  'ZAI',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 30,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 7,
                  ),
                ),
              ),
            ),
            const SizedBox(height: 30),
            const Text(
              'HOW CAN I ASSIST YOU?',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.white,
                fontSize: 24,
                fontWeight: FontWeight.bold,
                letterSpacing: 2,
              ),
            ),
            const SizedBox(height: 12),
            const Text(
              'ZAI Intelligence Core',
              style: TextStyle(
                color: Color(0xFF00D9FF),
                fontSize: 13,
                letterSpacing: 3,
              ),
            ),
            const SizedBox(height: 35),
            Wrap(
              alignment: WrapAlignment.center,
              spacing: 12,
              runSpacing: 12,
              children: [
                _suggestionCard(
                  Icons.computer,
                  'System status',
                  'Check my system',
                ),
                _suggestionCard(
                  Icons.code,
                  'Coding',
                  'Help me code',
                ),
                _suggestionCard(
                  Icons.lightbulb_outline,
                  'Ideas',
                  'Give me ideas',
                ),
                _suggestionCard(
                  Icons.auto_awesome,
                  'Analyze',
                  'Analyze something',
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _suggestionCard(
    IconData icon,
    String title,
    String command,
  ) {
    return InkWell(
      onTap: () => _quickCommand(command),
      borderRadius: BorderRadius.circular(14),
      child: Container(
        width: 190,
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          color: const Color(0xFF06121D),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: const Color(0xFF12364A),
          ),
        ),
        child: Row(
          children: [
            Icon(
              icon,
              color: const Color(0xFF00D9FF),
              size: 22,
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                title,
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMessages() {
    return ListView.builder(
      controller: _scrollController,
      padding: const EdgeInsets.all(28),
      itemCount: _messages.length + (_isTyping ? 1 : 0),
      itemBuilder: (BuildContext context, int index) {
        if (_isTyping && index == _messages.length) {
          return _typingIndicator();
        }

        return _messageBubble(_messages[index]);
      },
    );
  }

  Widget _messageBubble(_ChatMessage message) {
    final bool isUser = message.isUser;

    return Align(
      alignment:
          isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        constraints: const BoxConstraints(
          maxWidth: 750,
        ),
        margin: const EdgeInsets.only(bottom: 18),
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          color: isUser
              ? const Color(0xFF07354A)
              : const Color(0xFF07141F),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: isUser
                ? const Color(0xFF087C9D)
                : const Color(0xFF17364A),
          ),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(
              isUser
                  ? Icons.person_outline
                  : Icons.auto_awesome,
              color: isUser
                  ? const Color(0xFF8EB5C7)
                  : const Color(0xFF00D9FF),
              size: 22,
            ),
            const SizedBox(width: 14),
            Expanded(
              child: SelectableText(
                message.text,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 15,
                  height: 1.6,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _typingIndicator() {
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 18),
        padding: const EdgeInsets.symmetric(
          horizontal: 20,
          vertical: 14,
        ),
        decoration: BoxDecoration(
          color: const Color(0xFF07141F),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: const Color(0xFF17364A),
          ),
        ),
        child: const Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.auto_awesome,
              color: Color(0xFF00D9FF),
              size: 18,
            ),
            SizedBox(width: 10),
            Text(
              'ZAI is processing...',
              style: TextStyle(
                color: Color(0xFF8CA8B8),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildQuickCommands() {
    if (_messages.isEmpty) {
      return const SizedBox.shrink();
    }

    return Container(
      height: 55,
      padding: const EdgeInsets.symmetric(
        horizontal: 20,
      ),
      child: ListView(
        scrollDirection: Axis.horizontal,
        children: [
          _quickChip('Check system'),
          _quickChip('Open browser'),
          _quickChip('Analyze'),
          _quickChip('Create project'),
        ],
      ),
    );
  }

  Widget _quickChip(String text) {
    return Padding(
      padding: const EdgeInsets.only(right: 8),
      child: ActionChip(
        label: Text(text),
        onPressed: _isTyping
            ? null
            : () => _quickCommand(text),
        backgroundColor: const Color(0xFF071923),
        side: const BorderSide(
          color: Color(0xFF174257),
        ),
        labelStyle: const TextStyle(
          color: Color(0xFF8EB5C7),
        ),
      ),
    );
  }

  Widget _buildInput() {
    return Container(
      padding: const EdgeInsets.fromLTRB(
        20,
        12,
        20,
        20,
      ),
      decoration: const BoxDecoration(
        color: Color(0xFF030B14),
        border: Border(
          top: BorderSide(
            color: Color(0xFF123044),
          ),
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          IconButton(
            tooltip: 'Attach',
            onPressed: _isTyping ? null : () {},
            icon: const Icon(
              Icons.attach_file,
              color: Color(0xFF7592A3),
            ),
          ),
          Expanded(
            child: TextField(
              controller: _controller,
              enabled: !_isTyping,
              onSubmitted: (_) => _sendMessage(),
              keyboardType: TextInputType.multiline,
              textInputAction: TextInputAction.newline,
              style: const TextStyle(
                color: Colors.white,
              ),
              minLines: 1,
              maxLines: 4,
              decoration: InputDecoration(
                hintText: _isTyping
                    ? 'ZAI sedang memproses...'
                    : 'Ask ZAI anything...',
                hintStyle: const TextStyle(
                  color: Color(0xFF557181),
                ),
                filled: true,
                fillColor: const Color(0xFF07131E),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(16),
                  borderSide: const BorderSide(
                    color: Color(0xFF15384A),
                  ),
                ),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(16),
                  borderSide: const BorderSide(
                    color: Color(0xFF15384A),
                  ),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(16),
                  borderSide: const BorderSide(
                    color: Color(0xFF00D9FF),
                  ),
                ),
                disabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(16),
                  borderSide: const BorderSide(
                    color: Color(0xFF15384A),
                  ),
                ),
                contentPadding: const EdgeInsets.symmetric(
                  horizontal: 18,
                  vertical: 15,
                ),
              ),
            ),
          ),
          const SizedBox(width: 10),
          IconButton(
            tooltip: 'Voice',
            onPressed: _isTyping ? null : () {},
            icon: const Icon(
              Icons.mic_none,
              color: Color(0xFF00D9FF),
              size: 26,
            ),
          ),
          const SizedBox(width: 4),
          Container(
            decoration: BoxDecoration(
              color: _isTyping
                  ? const Color(0xFF155266)
                  : const Color(0xFF00AFCF),
              borderRadius: BorderRadius.circular(14),
              boxShadow: const [
                BoxShadow(
                  color: Color(0x5500D9FF),
                  blurRadius: 15,
                ),
              ],
            ),
            child: IconButton(
              tooltip: 'Send',
              onPressed: _isTyping ? null : _sendMessage,
              icon: const Icon(
                Icons.arrow_upward,
                color: Colors.white,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _ChatMessage {
  final String text;
  final bool isUser;

  const _ChatMessage({
    required this.text,
    required this.isUser,
  });
}