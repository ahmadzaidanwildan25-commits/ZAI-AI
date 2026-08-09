import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  // ============================================================
  // ZAI SERVER
  // ============================================================

  static const String _serverUrl = 'http://127.0.0.1:8000';

  // ============================================================
  // CONTROLLERS
  // ============================================================

  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  final List<_ChatMessage> _messages = <_ChatMessage>[];

  // ============================================================
  // STATE
  // ============================================================

  bool _isTyping = false;
  bool _serverOnline = false;

  // ============================================================
  // HTTP CLIENT
  // ============================================================

  final http.Client _httpClient = http.Client();

  // ============================================================
  // INIT
  // ============================================================

  @override
  void initState() {
    super.initState();

    _checkServer();
  }

  // ============================================================
  // DISPOSE
  // ============================================================

  @override
  void dispose() {
    _controller.dispose();
    _scrollController.dispose();
    _httpClient.close();

    super.dispose();
  }

  // ============================================================
  // CHECK ZAI SERVER
  // ============================================================

  Future<void> _checkServer() async {
    try {
      final http.Response response = await _httpClient
          .get(
            Uri.parse('$_serverUrl/health'),
          )
          .timeout(
            const Duration(seconds: 3),
          );

      if (!mounted) {
        return;
      }

      setState(() {
        _serverOnline = response.statusCode == 200;
      });
    } catch (_) {
      if (!mounted) {
        return;
      }

      setState(() {
        _serverOnline = false;
      });
    }
  }

  // ============================================================
  // SEND MESSAGE
  // ============================================================

  Future<void> _sendMessage() async {
    final String text = _controller.text.trim();

    if (text.isEmpty || _isTyping) {
      return;
    }

    // ----------------------------------------------------------
    // ADD USER MESSAGE
    // ----------------------------------------------------------

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

    // ----------------------------------------------------------
    // ADD EMPTY ZAI MESSAGE
    // ----------------------------------------------------------

    final int assistantIndex = _messages.length;

    setState(() {
      _messages.add(
        const _ChatMessage(
          text: '',
          isUser: false,
        ),
      );
    });

    _scrollToBottom();

    try {
      // --------------------------------------------------------
      // BUILD HISTORY
      // --------------------------------------------------------

      final List<Map<String, dynamic>> history =
          <Map<String, dynamic>>[];

      for (final _ChatMessage message in _messages) {
        if (message.isUser) {
          history.add(
            <String, dynamic>{
              'role': 'user',
              'content': message.text,
            },
          );
        } else if (message.text.isNotEmpty) {
          history.add(
            <String, dynamic>{
              'role': 'assistant',
              'content': message.text,
            },
          );
        }
      }

      // --------------------------------------------------------
      // REMOVE CURRENT EMPTY ASSISTANT FROM HISTORY
      // --------------------------------------------------------

      if (history.isNotEmpty &&
          history.last['role'] == 'assistant' &&
          history.last['content'] == '') {
        history.removeLast();
      }

      // --------------------------------------------------------
      // REQUEST
      // --------------------------------------------------------

      final http.Request request = http.Request(
        'POST',
        Uri.parse('$_serverUrl/chat'),
      );

      request.headers['Content-Type'] = 'application/json';
      request.headers['Accept'] = 'application/x-ndjson';

      request.body = jsonEncode(
        <String, dynamic>{
          'message': text,
          'history': history,
          'mode': 'auto',
        },
      );

      // --------------------------------------------------------
      // SEND STREAMING REQUEST
      // --------------------------------------------------------

      final http.StreamedResponse response =
          await _httpClient.send(request);

      if (response.statusCode != 200) {
        final String errorBody =
            await response.stream.bytesToString();

        throw Exception(
          'ZAI Server error ${response.statusCode}: $errorBody',
        );
      }

      // --------------------------------------------------------
      // READ STREAM
      // --------------------------------------------------------

      final Stream<String> lines = response.stream
          .transform(utf8.decoder)
          .transform(const LineSplitter());

      await for (final String line in lines) {
        if (line.trim().isEmpty) {
          continue;
        }

        Map<String, dynamic> data;

        try {
          data = jsonDecode(line) as Map<String, dynamic>;
        } catch (_) {
          continue;
        }

        final String type =
            data['type']?.toString() ?? '';

        // ------------------------------------------------------
        // START
        // ------------------------------------------------------

        if (type == 'start') {
          if (mounted) {
            setState(() {
              _serverOnline = true;
            });
          }

          continue;
        }

        // ------------------------------------------------------
        // TOKEN
        // ------------------------------------------------------

        if (type == 'token') {
          final String token =
              data['content']?.toString() ?? '';

          if (token.isEmpty) {
            continue;
          }

          if (!mounted) {
            continue;
          }

          setState(() {
            _messages[assistantIndex] = _ChatMessage(
              text: _messages[assistantIndex].text + token,
              isUser: false,
            );
          });

          _scrollToBottom();

          continue;
        }

        // ------------------------------------------------------
        // ERROR
        // ------------------------------------------------------

        if (type == 'error') {
          final String errorMessage =
              data['message']?.toString() ??
                  'ZAI mengalami kesalahan.';

          if (!mounted) {
            continue;
          }

          setState(() {
            _messages[assistantIndex] = _ChatMessage(
              text: 'ZAI ERROR:\n$errorMessage',
              isUser: false,
            );
          });

          continue;
        }

        // ------------------------------------------------------
        // DONE
        // ------------------------------------------------------

        if (type == 'done') {
          continue;
        }
      }
    } catch (error) {
      if (!mounted) {
        return;
      }

      setState(() {
        _serverOnline = false;

        _messages[assistantIndex] = _ChatMessage(
          text:
              'Tidak dapat terhubung ke ZAI Core.\n\n'
              'Pastikan FastAPI berjalan di:\n'
              '$_serverUrl\n\n'
              'Error:\n$error',
          isUser: false,
        );
      });
    } finally {
      if (!mounted) {
        return;
      }

      setState(() {
        _isTyping = false;
      });

      _scrollToBottom();
    }
  }

  // ============================================================
  // QUICK COMMAND
  // ============================================================

  void _sendQuickCommand(String command) {
    if (_isTyping) {
      return;
    }

    _controller.text = command;

    _sendMessage();
  }

  // ============================================================
  // SCROLL
  // ============================================================

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) {
        return;
      }

      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 120),
        curve: Curves.easeOut,
      );
    });
  }

  // ============================================================
  // BUILD
  // ============================================================

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF020911),
      body: SafeArea(
        child: Column(
          children: <Widget>[
            _buildHeader(),

            Expanded(
              child: _messages.isEmpty
                  ? _buildWelcome()
                  : _buildMessages(),
            ),

            if (_messages.isNotEmpty)
              _buildQuickCommands(),

            _buildInput(),
          ],
        ),
      ),
    );
  }

  // ============================================================
  // HEADER
  // ============================================================

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: 24,
        vertical: 18,
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
        children: <Widget>[
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
              size: 24,
            ),
          ),

          const SizedBox(width: 14),

          const Expanded(
            child: Column(
              crossAxisAlignment:
                  CrossAxisAlignment.start,
              children: <Widget>[
                Text(
                  'ZAI CHAT',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 21,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 2,
                  ),
                ),
                SizedBox(height: 4),
                Text(
                  'INTELLIGENCE CONVERSATION INTERFACE',
                  style: TextStyle(
                    color: Color(0xFF7190A5),
                    fontSize: 10,
                    letterSpacing: 1.3,
                  ),
                ),
              ],
            ),
          ),

          _buildOnlineIndicator(),
        ],
      ),
    );
  }

  // ============================================================
  // ONLINE INDICATOR
  // ============================================================

  Widget _buildOnlineIndicator() {
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: 12,
        vertical: 8,
      ),
      decoration: BoxDecoration(
        color: _serverOnline
            ? const Color(0xFF03231E)
            : const Color(0xFF241616),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: _serverOnline
              ? const Color(0xFF00D9A5)
              : const Color(0xFFFF5C5C),
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          Icon(
            Icons.circle,
            color: _serverOnline
                ? const Color(0xFF00E6A8)
                : const Color(0xFFFF5C5C),
            size: 8,
          ),

          const SizedBox(width: 7),

          Text(
            _serverOnline ? 'ONLINE' : 'OFFLINE',
            style: TextStyle(
              color: _serverOnline
                  ? const Color(0xFF00E6A8)
                  : const Color(0xFFFF5C5C),
              fontSize: 10,
              fontWeight: FontWeight.bold,
              letterSpacing: 1,
            ),
          ),
        ],
      ),
    );
  }

  // ============================================================
  // WELCOME
  // ============================================================

  Widget _buildWelcome() {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(28),
        child: Column(
          mainAxisAlignment:
              MainAxisAlignment.center,
          children: <Widget>[
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
                boxShadow: const <BoxShadow>[
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

            const SizedBox(height: 28),

            const Text(
              'HOW CAN I ASSIST YOU?',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.white,
                fontSize: 23,
                fontWeight: FontWeight.bold,
                letterSpacing: 1.5,
              ),
            ),

            const SizedBox(height: 10),

            const Text(
              'ZAI Intelligence Core',
              style: TextStyle(
                color: Color(0xFF00D9FF),
                fontSize: 13,
                letterSpacing: 2.5,
              ),
            ),

            const SizedBox(height: 30),

            Wrap(
              alignment: WrapAlignment.center,
              spacing: 12,
              runSpacing: 12,
              children: <Widget>[
                _buildSuggestionCard(
                  icon: Icons.computer,
                  title: 'System status',
                  command: 'Check my system',
                ),
                _buildSuggestionCard(
                  icon: Icons.code,
                  title: 'Coding',
                  command: 'Help me code',
                ),
                _buildSuggestionCard(
                  icon: Icons.lightbulb_outline,
                  title: 'Ideas',
                  command: 'Give me ideas',
                ),
                _buildSuggestionCard(
                  icon: Icons.auto_awesome,
                  title: 'Analyze',
                  command: 'Analyze something',
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  // ============================================================
  // SUGGESTION CARD
  // ============================================================

  Widget _buildSuggestionCard({
    required IconData icon,
    required String title,
    required String command,
  }) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: () => _sendQuickCommand(command),
        borderRadius: BorderRadius.circular(14),
        child: Container(
          width: 190,
          padding: const EdgeInsets.all(17),
          decoration: BoxDecoration(
            color: const Color(0xFF06121D),
            borderRadius: BorderRadius.circular(14),
            border: Border.all(
              color: const Color(0xFF12364A),
            ),
          ),
          child: Row(
            children: <Widget>[
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
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ============================================================
  // MESSAGES
  // ============================================================

  Widget _buildMessages() {
    return ListView.builder(
      controller: _scrollController,
      padding: const EdgeInsets.all(24),
      itemCount: _messages.length +
          (_isTyping ? 1 : 0),
      itemBuilder:
          (BuildContext context, int index) {
        if (_isTyping &&
            index == _messages.length) {
          return _buildTypingIndicator();
        }

        return _buildMessageBubble(
          _messages[index],
        );
      },
    );
  }

  // ============================================================
  // MESSAGE BUBBLE
  // ============================================================

  Widget _buildMessageBubble(
    _ChatMessage message,
  ) {
    final bool isUser = message.isUser;

    return Align(
      alignment: isUser
          ? Alignment.centerRight
          : Alignment.centerLeft,
      child: Container(
        constraints: const BoxConstraints(
          maxWidth: 760,
        ),
        margin: const EdgeInsets.only(
          bottom: 16,
        ),
        padding: const EdgeInsets.all(17),
        decoration: BoxDecoration(
          color: isUser
              ? const Color(0xFF07354A)
              : const Color(0xFF07141F),
          borderRadius:
              BorderRadius.circular(16),
          border: Border.all(
            color: isUser
                ? const Color(0xFF087C9D)
                : const Color(0xFF17364A),
          ),
        ),
        child: Row(
          crossAxisAlignment:
              CrossAxisAlignment.start,
          children: <Widget>[
            Icon(
              isUser
                  ? Icons.person_outline
                  : Icons.auto_awesome,
              color: isUser
                  ? const Color(0xFF8EB5C7)
                  : const Color(0xFF00D9FF),
              size: 21,
            ),

            const SizedBox(width: 13),

            Expanded(
              child: SelectableText(
                message.text.isEmpty
                    ? 'ZAI is processing...'
                    : message.text,
                style: TextStyle(
                  color: message.text.isEmpty
                      ? const Color(0xFF8CA8B8)
                      : Colors.white,
                  fontSize: 15,
                  height: 1.5,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ============================================================
  // TYPING
  // ============================================================

  Widget _buildTypingIndicator() {
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(
          bottom: 16,
        ),
        padding: const EdgeInsets.symmetric(
          horizontal: 18,
          vertical: 13,
        ),
        decoration: BoxDecoration(
          color: const Color(0xFF07141F),
          borderRadius:
              BorderRadius.circular(14),
          border: Border.all(
            color: const Color(0xFF17364A),
          ),
        ),
        child: const Row(
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            SizedBox(
              width: 15,
              height: 15,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                color: Color(0xFF00D9FF),
              ),
            ),

            SizedBox(width: 10),

            Text(
              'ZAI is thinking...',
              style: TextStyle(
                color: Color(0xFF8CA8B8),
                fontSize: 13,
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ============================================================
  // QUICK COMMANDS
  // ============================================================

  Widget _buildQuickCommands() {
    return SizedBox(
      height: 52,
      child: ListView(
        padding:
            const EdgeInsets.symmetric(
          horizontal: 18,
        ),
        scrollDirection: Axis.horizontal,
        children: <Widget>[
          _buildQuickChip('Check system'),
          _buildQuickChip('Open browser'),
          _buildQuickChip('Analyze'),
          _buildQuickChip('Create project'),
        ],
      ),
    );
  }

  Widget _buildQuickChip(String text) {
    return Padding(
      padding:
          const EdgeInsets.only(right: 8),
      child: ActionChip(
        label: Text(text),
        onPressed: () =>
            _sendQuickCommand(text),
        backgroundColor:
            const Color(0xFF071923),
        side: const BorderSide(
          color: Color(0xFF174257),
        ),
        labelStyle: const TextStyle(
          color: Color(0xFF8EB5C7),
          fontSize: 12,
        ),
      ),
    );
  }

  // ============================================================
  // INPUT
  // ============================================================

  Widget _buildInput() {
    return Container(
      padding: const EdgeInsets.fromLTRB(
        16,
        10,
        16,
        16,
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
        crossAxisAlignment:
            CrossAxisAlignment.end,
        children: <Widget>[
          IconButton(
            tooltip: 'Attach',
            onPressed: _isTyping
                ? null
                : () {},
            icon: const Icon(
              Icons.attach_file,
              color: Color(0xFF7592A3),
            ),
          ),

          Expanded(
            child: TextField(
              controller: _controller,
              enabled: !_isTyping,
              onSubmitted: (_) =>
                  _sendMessage(),
              minLines: 1,
              maxLines: 4,
              keyboardType:
                  TextInputType.multiline,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 15,
              ),
              decoration:
                  InputDecoration(
                hintText:
                    'Ask ZAI anything...',
                hintStyle:
                    const TextStyle(
                  color: Color(0xFF557181),
                ),
                filled: true,
                fillColor:
                    const Color(0xFF07131E),
                contentPadding:
                    const EdgeInsets.symmetric(
                  horizontal: 18,
                  vertical: 14,
                ),
                border:
                    OutlineInputBorder(
                  borderRadius:
                      BorderRadius.circular(
                    16,
                  ),
                  borderSide:
                      const BorderSide(
                    color: Color(0xFF15384A),
                  ),
                ),
                enabledBorder:
                    OutlineInputBorder(
                  borderRadius:
                      BorderRadius.circular(
                    16,
                  ),
                  borderSide:
                      const BorderSide(
                    color: Color(0xFF15384A),
                  ),
                ),
                focusedBorder:
                    OutlineInputBorder(
                  borderRadius:
                      BorderRadius.circular(
                    16,
                  ),
                  borderSide:
                      const BorderSide(
                    color: Color(0xFF00D9FF),
                  ),
                ),
              ),
            ),
          ),

          const SizedBox(width: 8),

          IconButton(
            tooltip: 'Voice',
            onPressed:
                _isTyping ? null : () {},
            icon: const Icon(
              Icons.mic_none,
              color: Color(0xFF00D9FF),
              size: 25,
            ),
          ),

          const SizedBox(width: 3),

          Container(
            decoration: BoxDecoration(
              color: _isTyping
                  ? const Color(0xFF174257)
                  : const Color(0xFF00AFCF),
              borderRadius:
                  BorderRadius.circular(14),
              boxShadow:
                  const <BoxShadow>[
                BoxShadow(
                  color: Color(0x5500D9FF),
                  blurRadius: 15,
                ),
              ],
            ),
            child: IconButton(
              tooltip: 'Send',
              onPressed: _isTyping
                  ? null
                  : _sendMessage,
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

// ============================================================
// CHAT MESSAGE MODEL
// ============================================================

class _ChatMessage {
  final String text;
  final bool isUser;

  const _ChatMessage({
    required this.text,
    required this.isUser,
  });
}