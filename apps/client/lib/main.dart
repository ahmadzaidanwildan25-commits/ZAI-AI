import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ZAIApp());
}

// ============================================================
// ZAI APP
// ============================================================

class ZAIApp extends StatelessWidget {
  const ZAIApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'ZAI',
      theme: ZAITheme.dark(),
      home: const ZAIShell(),
    );
  }
}

// ============================================================
// THEME
// ============================================================

class ZAITheme {
  static const background = Color(0xFF03070D);
  static const surface = Color(0xFF08111C);
  static const surface2 = Color(0xFF0C1724);
  static const cyan = Color(0xFF00E5FF);
  static const blue = Color(0xFF1976FF);
  static const green = Color(0xFF00F5A0);
  static const text = Color(0xFFE8F7FF);
  static const muted = Color(0xFF7891A5);
  static const border = Color(0xFF153044);

  static ThemeData dark() {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: background,
      fontFamily: 'Segoe UI',
      colorScheme: const ColorScheme.dark(
        primary: cyan,
        secondary: blue,
        surface: surface,
      ),
    );
  }
}

// ============================================================
// NAVIGATION ITEM
// ============================================================

class ZAINavItem {
  final IconData icon;
  final String title;

  const ZAINavItem(this.icon, this.title);
}

const navItems = <ZAINavItem>[
  ZAINavItem(Icons.dashboard_rounded, 'Dashboard'),
  ZAINavItem(Icons.chat_bubble_rounded, 'Chat'),
  ZAINavItem(Icons.mic_rounded, 'Voice'),
  ZAINavItem(Icons.memory_rounded, 'Memory'),
  ZAINavItem(Icons.smart_toy_rounded, 'Agents'),
  ZAINavItem(Icons.devices_rounded, 'Devices'),
  ZAINavItem(Icons.account_tree_rounded, 'Automation'),
  ZAINavItem(Icons.folder_rounded, 'Projects'),
  ZAINavItem(Icons.menu_book_rounded, 'Knowledge'),
  ZAINavItem(Icons.auto_awesome_rounded, 'Evolution'),
  ZAINavItem(Icons.analytics_rounded, 'Analytics'),
  ZAINavItem(Icons.settings_rounded, 'Settings'),
];

// ============================================================
// MAIN SHELL
// ============================================================

class ZAIShell extends StatefulWidget {
  const ZAIShell({super.key});

  @override
  State<ZAIShell> createState() => _ZAIShellState();
}

class _ZAIShellState extends State<ZAIShell> {
  int selectedIndex = 0;
  bool sidebarCollapsed = false;

  final TextEditingController commandController = TextEditingController();

  @override
  void dispose() {
    commandController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: LayoutBuilder(
        builder: (context, constraints) {
          final mobile = constraints.maxWidth < 850;

          return Stack(
            children: [
              const ZAIBackground(),

              Column(
                children: [
                  ZAITopBar(
                    onMenu: () {
                      if (mobile) {
                        _showMobileMenu(context);
                      } else {
                        setState(() {
                          sidebarCollapsed = !sidebarCollapsed;
                        });
                      }
                    },
                  ),

                  Expanded(
                    child: Row(
                      children: [
                        if (!mobile)
                          ZAISidebar(
                            collapsed: sidebarCollapsed,
                            selectedIndex: selectedIndex,
                            onSelected: (index) {
                              setState(() {
                                selectedIndex = index;
                              });
                            },
                          ),

                        Expanded(
                          child: _buildPage(selectedIndex),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _buildPage(int index) {
    switch (index) {
      case 0:
        return const ZAIDashboard();

      case 1:
        return const ZAIChatPage();

      case 2:
        return const ZAIVoicePage();

      case 3:
        return const ZAIMemoryPage();

      case 4:
        return const ZAIAgentsPage();
      case 5:
        return const ZAIDevicesPage();

      case 6:
        return const ZAIAutomationPage();

      case 7:
        return const ZAIProjectsPage();

      case 8:
        return const ZAIKnowledgePage();

      case 9:
        return const ZAIEvolutionPage();

      case 10:
        return const ZAIAnalyticsPage();

      case 11:
        return const ZAISettingsPage();

      default:
        return const ZAIDashboard();
    }
  }

  void _showMobileMenu(BuildContext context) {
    showModalBottomSheet(
      context: context,
      backgroundColor: ZAITheme.surface,
      isScrollControlled: true,
      builder: (context) {
        return SafeArea(
          child: ListView.builder(
            shrinkWrap: true,
            itemCount: navItems.length,
            itemBuilder: (context, index) {
              final item = navItems[index];

              return ListTile(
                leading: Icon(
                  item.icon,
                  color: selectedIndex == index
                      ? ZAITheme.cyan
                      : ZAITheme.muted,
                ),
                title: Text(item.title),
                onTap: () {
                  Navigator.pop(context);

                  setState(() {
                    selectedIndex = index;
                  });
                },
              );
            },
          ),
        );
      },
    );
  }
}

// ============================================================
// TOP BAR
// ============================================================

class ZAITopBar extends StatelessWidget {
  final VoidCallback onMenu;

  const ZAITopBar({
    super.key,
    required this.onMenu,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 68,
      decoration: BoxDecoration(
        color: const Color(0xDD03070D),
        border: const Border(
          bottom: BorderSide(
            color: ZAITheme.border,
            width: 1,
          ),
        ),
      ),
      child: Row(
        children: [
          const SizedBox(width: 18),

          IconButton(
            onPressed: onMenu,
            icon: const Icon(Icons.menu_rounded),
          ),

          const SizedBox(width: 10),

          const Text(
            'ZAI',
            style: TextStyle(
              fontSize: 23,
              fontWeight: FontWeight.w800,
              letterSpacing: 4,
              color: ZAITheme.text,
            ),
          ),

          const SizedBox(width: 14),

          Container(
            padding: const EdgeInsets.symmetric(
              horizontal: 10,
              vertical: 5,
            ),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(20),
              color: ZAITheme.green.withValues(alpha: .08),
              border: Border.all(
                color: ZAITheme.green.withValues(alpha: .25),
              ),
            ),
            child: const Row(
              children: [
                _StatusDot(),
                SizedBox(width: 7),
                Text(
                  'SYSTEM ONLINE',
                  style: TextStyle(
                    color: ZAITheme.green,
                    fontSize: 10,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 1.1,
                  ),
                ),
              ],
            ),
          ),

          const Spacer(),

          const ZAIClock(),

          const SizedBox(width: 10),

          IconButton(
            onPressed: () {},
            icon: const Icon(Icons.notifications_none_rounded),
          ),

          IconButton(
            onPressed: () {},
            icon: const Icon(Icons.settings_outlined),
          ),

          const SizedBox(width: 14),
        ],
      ),
    );
  }
}

class _StatusDot extends StatefulWidget {
  const _StatusDot();

  @override
  State<_StatusDot> createState() => _StatusDotState();
}

class _StatusDotState extends State<_StatusDot>
    with SingleTickerProviderStateMixin {
  late AnimationController controller;

  @override
  void initState() {
    super.initState();

    controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 1),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: controller,
      builder: (_, child) {
        return Opacity(
          opacity: .45 + controller.value * .55,
          child: child,
        );
      },
      child: Container(
        width: 7,
        height: 7,
        decoration: const BoxDecoration(
          shape: BoxShape.circle,
          color: ZAITheme.green,
        ),
      ),
    );
  }
}

// ============================================================
// CLOCK
// ============================================================

class ZAIClock extends StatefulWidget {
  const ZAIClock({super.key});

  @override
  State<ZAIClock> createState() => _ZAIClockState();
}

class _ZAIClockState extends State<ZAIClock> {
  late Timer timer;
  DateTime now = DateTime.now();

  @override
  void initState() {
    super.initState();

    timer = Timer.periodic(
      const Duration(seconds: 1),
      (_) {
        if (mounted) {
          setState(() {
            now = DateTime.now();
          });
        }
      },
    );
  }

  @override
  void dispose() {
    timer.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final time =
        '${now.hour.toString().padLeft(2, '0')}:'
        '${now.minute.toString().padLeft(2, '0')}:'
        '${now.second.toString().padLeft(2, '0')}';

    return Text(
      time,
      style: const TextStyle(
        color: ZAITheme.muted,
        fontSize: 12,
        fontFeatures: [
          FontFeature.tabularFigures(),
        ],
      ),
    );
  }
}

// ============================================================
// SIDEBAR
// ============================================================

class ZAISidebar extends StatelessWidget {
  final bool collapsed;
  final int selectedIndex;
  final ValueChanged<int> onSelected;

  const ZAISidebar({
    super.key,
    required this.collapsed,
    required this.selectedIndex,
    required this.onSelected,
  });

  @override
  Widget build(BuildContext context) {
    final width = collapsed ? 78.0 : 220.0;

    return AnimatedContainer(
      duration: const Duration(milliseconds: 250),
      width: width,
      decoration: const BoxDecoration(
        color: Color(0xCC050B13),
        border: Border(
          right: BorderSide(
            color: ZAITheme.border,
          ),
        ),
      ),
      child: Column(
        children: [
          const SizedBox(height: 16),

          ZAINavHeader(
            collapsed: collapsed,
          ),

          const SizedBox(height: 14),

          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.symmetric(
                horizontal: 10,
              ),
              itemCount: navItems.length,
              itemBuilder: (context, index) {
                final item = navItems[index];

                return ZAINavButton(
                  item: item,
                  selected: selectedIndex == index,
                  collapsed: collapsed,
                  onTap: () => onSelected(index),
                );
              },
            ),
          ),

          Container(
            margin: const EdgeInsets.all(12),
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(14),
              color: ZAITheme.green.withValues(alpha: .035),
              border: Border.all(
                color: ZAITheme.green.withValues(alpha: .12),
              ),
            ),
            child: collapsed
                ? const Icon(
                    Icons.shield_outlined,
                    color: ZAITheme.green,
                  )
                : const Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Icon(
                            Icons.shield_outlined,
                            size: 17,
                            color: ZAITheme.green,
                          ),
                          SizedBox(width: 8),
                          Text(
                            'SECURITY',
                            style: TextStyle(
                              color: ZAITheme.green,
                              fontSize: 10,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                      SizedBox(height: 8),
                      Text(
                        'Protected',
                        style: TextStyle(
                          color: ZAITheme.text,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
          ),
        ],
      ),
    );
  }
}

class ZAINavHeader extends StatelessWidget {
  final bool collapsed;

  const ZAINavHeader({
    super.key,
    required this.collapsed,
  });

  @override
  Widget build(BuildContext context) {
    if (collapsed) {
      return const ZAIOrbIcon(size: 42);
    }

    return const Row(
      children: [
        SizedBox(width: 12),
        ZAIOrbIcon(size: 38),
        SizedBox(width: 12),
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'ZAI',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 16,
                letterSpacing: 2,
              ),
            ),
            Text(
              'AI COMMAND CENTER',
              style: TextStyle(
                color: ZAITheme.muted,
                fontSize: 8,
                letterSpacing: 1,
              ),
            ),
          ],
        ),
      ],
    );
  }
}

class ZAINavButton extends StatelessWidget {
  final ZAINavItem item;
  final bool selected;
  final bool collapsed;
  final VoidCallback onTap;

  const ZAINavButton({
    super.key,
    required this.item,
    required this.selected,
    required this.collapsed,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: collapsed ? item.title : '',
      child: Container(
        margin: const EdgeInsets.only(bottom: 5),
        child: Material(
          color: Colors.transparent,
          child: InkWell(
            borderRadius: BorderRadius.circular(12),
            onTap: onTap,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 180),
              height: 44,
              padding: EdgeInsets.symmetric(
                horizontal: collapsed ? 0 : 13,
              ),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(12),
                color: selected
                    ? ZAITheme.cyan.withValues(alpha: .08)
                    : Colors.transparent,
                border: Border.all(
                  color: selected
                      ? ZAITheme.cyan.withValues(alpha: .25)
                      : Colors.transparent,
                ),
              ),
              child: Row(
                mainAxisAlignment: collapsed
                    ? MainAxisAlignment.center
                    : MainAxisAlignment.start,
                children: [
                  Icon(
                    item.icon,
                    size: 19,
                    color: selected
                        ? ZAITheme.cyan
                        : ZAITheme.muted,
                  ),

                  if (!collapsed) ...[
                    const SizedBox(width: 12),
                    Text(
                      item.title,
                      style: TextStyle(
                        fontSize: 12,
                        color: selected
                            ? ZAITheme.text
                            : ZAITheme.muted,
                        fontWeight: selected
                            ? FontWeight.w600
                            : FontWeight.w400,
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ============================================================
// BACKGROUND
// ============================================================

class ZAIBackground extends StatelessWidget {
  const ZAIBackground({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        gradient: RadialGradient(
          center: Alignment.topRight,
          radius: 1.2,
          colors: [
            Color(0xFF071B2B),
            ZAITheme.background,
          ],
        ),
      ),
    );
  }
}

// ============================================================
// DASHBOARD
// ============================================================

class ZAIDashboard extends StatelessWidget {
  const ZAIDashboard({super.key});

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final narrow = constraints.maxWidth < 1100;

        return SingleChildScrollView(
          padding: const EdgeInsets.all(22),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const ZAIPageHeader(
                title: 'Command Center',
                subtitle:
                    'Central intelligence and system overview',
                icon: Icons.dashboard_rounded,
              ),

              const SizedBox(height: 20),

              if (narrow)
                const Column(
                  children: [
                    ZAIHeroCore(),
                    SizedBox(height: 16),
                    ZAISystemPanel(),
                  ],
                )
              else
                const Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      flex: 2,
                      child: ZAIHeroCore(),
                    ),
                    SizedBox(width: 16),
                    Expanded(
                      child: ZAISystemPanel(),
                    ),
                  ],
                ),

              const SizedBox(height: 16),

              const ZAIStatsRow(),

              const SizedBox(height: 16),

              GridView.count(
                crossAxisCount: narrow ? 1 : 3,
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                mainAxisSpacing: 16,
                crossAxisSpacing: 16,
                childAspectRatio: narrow ? 2.7 : 1.55,
                children: const [
                  ZAIActivityCard(),
                  ZAIDeviceStatusCard(),
                  ZAITaskCard(),
                ],
              ),
            ],
          ),
        );
      },
    );
  }
}

// ============================================================
// PAGE HEADER
// ============================================================

class ZAIPageHeader extends StatelessWidget {
  final String title;
  final String subtitle;
  final IconData icon;

  const ZAIPageHeader({
    super.key,
    required this.title,
    required this.subtitle,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(
          width: 42,
          height: 42,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(12),
            color: ZAITheme.cyan.withValues(alpha: .07),
            border: Border.all(
              color: ZAITheme.cyan.withValues(alpha: .18),
            ),
          ),
          child: Icon(
            icon,
            color: ZAITheme.cyan,
            size: 20,
          ),
        ),
        const SizedBox(width: 13),
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: const TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 3),
            Text(
              subtitle,
              style: const TextStyle(
                color: ZAITheme.muted,
                fontSize: 11,
              ),
            ),
          ],
        ),
      ],
    );
  }
}

// ============================================================
// HERO CORE
// ============================================================

class ZAIHeroCore extends StatefulWidget {
  const ZAIHeroCore({super.key});

  @override
  State<ZAIHeroCore> createState() => _ZAIHeroCoreState();
}

class _ZAIHeroCoreState extends State<ZAIHeroCore>
    with SingleTickerProviderStateMixin {
  late AnimationController controller;

  @override
  void initState() {
    super.initState();

    controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 7),
    )..repeat();
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ZAIGlassCard(
      padding: const EdgeInsets.all(20),
      child: Column(
        children: [
          Row(
            children: [
              const Text(
                'ZAI CORE',
                style: TextStyle(
                  fontSize: 11,
                  letterSpacing: 2,
                  fontWeight: FontWeight.bold,
                  color: ZAITheme.muted,
                ),
              ),
              const Spacer(),
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 8,
                  vertical: 4,
                ),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(20),
                  color: ZAITheme.green.withValues(alpha: .06),
                ),
                child: const Text(
                  'READY',
                  style: TextStyle(
                    color: ZAITheme.green,
                    fontSize: 9,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),

          const SizedBox(height: 8),

          SizedBox(
            height: 310,
            child: AnimatedBuilder(
              animation: controller,
              builder: (context, child) {
                return CustomPaint(
                  painter: ZAICorePainter(
                    animation: controller.value,
                  ),
                  child: const Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          'ZAI',
                          style: TextStyle(
                            fontSize: 38,
                            fontWeight: FontWeight.w800,
                            letterSpacing: 7,
                            color: ZAITheme.text,
                          ),
                        ),
                        SizedBox(height: 6),
                        Text(
                          'INTELLIGENCE CORE',
                          style: TextStyle(
                            color: ZAITheme.cyan,
                            fontSize: 9,
                            letterSpacing: 3,
                          ),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),

          const SizedBox(height: 4),

          const Text(
            'How can I assist you?',
            style: TextStyle(
              color: ZAITheme.muted,
              fontSize: 13,
            ),
          ),
        ],
      ),
    );
  }
}

// ============================================================
// CORE PAINTER
// ============================================================

class ZAICorePainter extends CustomPainter {
  final double animation;

  ZAICorePainter({
    required this.animation,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(
      size.width / 2,
      size.height / 2,
    );

    final shortest = math.min(
      size.width,
      size.height,
    );

    final radius = shortest * .28;

    final glow = Paint()
      ..color = ZAITheme.cyan.withValues(alpha: .035)
      ..maskFilter = const MaskFilter.blur(
        BlurStyle.normal,
        35,
      );

    canvas.drawCircle(
      center,
      radius * 1.15,
      glow,
    );

    for (int i = 0; i < 3; i++) {
      final r = radius + i * 22;

      final paint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = i == 0 ? 1.8 : 1
        ..color = ZAITheme.cyan.withValues(
          alpha: i == 0 ? .45 : .15,
        );

      canvas.drawCircle(
        center,
        r,
        paint,
      );
    }

    final orbitPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2
      ..color = ZAITheme.blue.withValues(alpha: .4);

    final rect = Rect.fromCircle(
      center: center,
      radius: radius + 43,
    );

    canvas.drawArc(
      rect,
      animation * math.pi * 2,
      math.pi * .85,
      false,
      orbitPaint,
    );

    canvas.drawArc(
      rect,
      -animation * math.pi * 2,
      math.pi * .45,
      false,
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2
        ..color = ZAITheme.cyan.withValues(alpha: .8),
    );

    for (int i = 0; i < 12; i++) {
      final angle =
          animation * math.pi * 2 +
          i * math.pi * 2 / 12;

      final x =
          center.dx + math.cos(angle) * (radius + 66);

      final y =
          center.dy + math.sin(angle) * (radius + 66);

      canvas.drawCircle(
        Offset(x, y),
        i % 3 == 0 ? 2.2 : 1,
        Paint()
          ..color = ZAITheme.cyan.withValues(
            alpha: i % 3 == 0 ? .8 : .25,
          ),
      );
    }
  }

  @override
  bool shouldRepaint(covariant ZAICorePainter oldDelegate) {
    return oldDelegate.animation != animation;
  }
}

// ============================================================
// SYSTEM PANEL
// ============================================================

class ZAISystemPanel extends StatelessWidget {
  const ZAISystemPanel({super.key});

  @override
  Widget build(BuildContext context) {
    return ZAIGlassCard(
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          children: [
            const _PanelTitle(
              title: 'SYSTEM STATUS',
              icon: Icons.monitor_heart_outlined,
            ),
            const SizedBox(height: 18),
            ZAIProgressMetric(
              title: 'CPU',
              value: '23%',
              progress: .23,
            ),
            ZAIProgressMetric(
              title: 'MEMORY',
              value: '41%',
              progress: .41,
            ),
            ZAIProgressMetric(
              title: 'GPU',
              value: '12%',
              progress: .12,
            ),
            ZAIProgressMetric(
              title: 'NETWORK',
              value: '8 ms',
              progress: .08,
            ),
            const SizedBox(height: 8),
            const Divider(
              color: ZAITheme.border,
            ),
            const SizedBox(height: 8),
            const ZAIStatusLine(
              title: 'AI CORE',
              value: 'ONLINE',
            ),
            const ZAIStatusLine(
              title: 'VOICE',
              value: 'READY',
            ),
            const ZAIStatusLine(
              title: 'MEMORY',
              value: 'READY',
            ),
            const ZAIStatusLine(
              title: 'SECURITY',
              value: 'PROTECTED',
            ),
          ],
        ),
      ),
    );
  }
}

class ZAIProgressMetric extends StatelessWidget {
  final String title;
  final String value;
  final double progress;

  const ZAIProgressMetric({
    super.key,
    required this.title,
    required this.value,
    required this.progress,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 15),
      child: Column(
        children: [
          Row(
            children: [
              Text(
                title,
                style: const TextStyle(
                  fontSize: 10,
                  color: ZAITheme.muted,
                  letterSpacing: 1,
                ),
              ),
              const Spacer(),
              Text(
                value,
                style: const TextStyle(
                  fontSize: 10,
                  color: ZAITheme.text,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 7),
          ClipRRect(
            borderRadius: BorderRadius.circular(20),
            child: LinearProgressIndicator(
              value: progress,
              minHeight: 3,
              backgroundColor: ZAITheme.border,
              color: ZAITheme.cyan,
            ),
          ),
        ],
      ),
    );
  }
}

class ZAIStatusLine extends StatelessWidget {
  final String title;
  final String value;

  const ZAIStatusLine({
    super.key,
    required this.title,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        children: [
          const Icon(
            Icons.circle,
            size: 6,
            color: ZAITheme.green,
          ),
          const SizedBox(width: 8),
          Text(
            title,
            style: const TextStyle(
              color: ZAITheme.muted,
              fontSize: 10,
            ),
          ),
          const Spacer(),
          Text(
            value,
            style: const TextStyle(
              color: ZAITheme.green,
              fontSize: 9,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
}

// ============================================================
// STATS
// ============================================================

class ZAIStatsRow extends StatelessWidget {
  const ZAIStatsRow({super.key});

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final columns = constraints.maxWidth < 700 ? 2 : 4;

        return GridView.count(
          crossAxisCount: columns,
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          crossAxisSpacing: 12,
          mainAxisSpacing: 12,
          childAspectRatio: 2.5,
          children: const [
            ZAIStatCard(
              icon: Icons.memory,
              title: 'MEMORY',
              value: '2,481',
              subtitle: 'records',
            ),
            ZAIStatCard(
              icon: Icons.smart_toy_outlined,
              title: 'AGENTS',
              value: '7',
              subtitle: 'available',
            ),
            ZAIStatCard(
              icon: Icons.devices_outlined,
              title: 'DEVICES',
              value: '2',
              subtitle: 'connected',
            ),
            ZAIStatCard(
              icon: Icons.auto_awesome,
              title: 'EVOLUTION',
              value: 'v0.1.0',
              subtitle: 'stable',
            ),
          ],
        );
      },
    );
  }
}

class ZAIStatCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String value;
  final String subtitle;

  const ZAIStatCard({
    super.key,
    required this.icon,
    required this.title,
    required this.value,
    required this.subtitle,
  });

  @override
  Widget build(BuildContext context) {
    return ZAIGlassCard(
      child: Row(
        children: [
          Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(11),
              color: ZAITheme.cyan.withValues(alpha: .06),
            ),
            child: Icon(
              icon,
              size: 18,
              color: ZAITheme.cyan,
            ),
          ),
          const SizedBox(width: 11),
          Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: const TextStyle(
                  fontSize: 8,
                  color: ZAITheme.muted,
                  letterSpacing: 1,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                value,
                style: const TextStyle(
                  fontSize: 17,
                  fontWeight: FontWeight.bold,
                ),
              ),
              Text(
                subtitle,
                style: const TextStyle(
                  fontSize: 8,
                  color: ZAITheme.muted,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ============================================================
// SMALL DASHBOARD CARDS
// ============================================================

class ZAIActivityCard extends StatelessWidget {
  const ZAIActivityCard({super.key});

  @override
  Widget build(BuildContext context) {
    return const ZAIInfoCard(
      title: 'RECENT ACTIVITY',
      icon: Icons.bolt_rounded,
      children: [
        ZAIActivityLine(
          text: 'System initialized',
          time: 'now',
        ),
        ZAIActivityLine(
          text: 'ZAI Core online',
          time: '1m',
        ),
        ZAIActivityLine(
          text: 'Memory engine ready',
          time: '2m',
        ),
        ZAIActivityLine(
          text: 'Security check passed',
          time: '3m',
        ),
      ],
    );
  }
}

class ZAIDeviceStatusCard extends StatelessWidget {
  const ZAIDeviceStatusCard({super.key});

  @override
  Widget build(BuildContext context) {
    return const ZAIInfoCard(
      title: 'CONNECTED DEVICES',
      icon: Icons.devices_rounded,
      children: [
        ZAIActivityLine(
          text: 'Windows PC',
          time: 'ONLINE',
          green: true,
        ),
        ZAIActivityLine(
          text: 'Android',
          time: 'READY',
          green: true,
        ),
        ZAIActivityLine(
          text: 'Smart Home',
          time: 'WAITING',
        ),
      ],
    );
  }
}

class ZAITaskCard extends StatelessWidget {
  const ZAITaskCard({super.key});

  @override
  Widget build(BuildContext context) {
    return const ZAIInfoCard(
      title: 'CURRENT TASK',
      icon: Icons.task_alt_rounded,
      children: [
        Text(
          'Waiting for command...',
          style: TextStyle(
            color: ZAITheme.text,
            fontSize: 13,
          ),
        ),
        SizedBox(height: 12),
        LinearProgressIndicator(
          value: 0,
          minHeight: 3,
          backgroundColor: ZAITheme.border,
          color: ZAITheme.cyan,
        ),
        SizedBox(height: 9),
        Text(
          'No active task',
          style: TextStyle(
            color: ZAITheme.muted,
            fontSize: 9,
          ),
        ),
      ],
    );
  }
}

class ZAIInfoCard extends StatelessWidget {
  final String title;
  final IconData icon;
  final List<Widget> children;

  const ZAIInfoCard({
    super.key,
    required this.title,
    required this.icon,
    required this.children,
  });

  @override
  Widget build(BuildContext context) {
    return ZAIGlassCard(
      child: Padding(
        padding: const EdgeInsets.all(17),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _PanelTitle(
              title: title,
              icon: icon,
            ),
            const SizedBox(height: 13),
            ...children,
          ],
        ),
      ),
    );
  }
}

class ZAIActivityLine extends StatelessWidget {
  final String text;
  final String time;
  final bool green;

  const ZAIActivityLine({
    super.key,
    required this.text,
    required this.time,
    this.green = false,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        children: [
          Icon(
            Icons.circle,
            size: 5,
            color: green
                ? ZAITheme.green
                : ZAITheme.cyan,
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              text,
              style: const TextStyle(
                fontSize: 10,
                color: ZAITheme.text,
              ),
            ),
          ),
          Text(
            time,
            style: TextStyle(
              fontSize: 8,
              color: green
                  ? ZAITheme.green
                  : ZAITheme.muted,
            ),
          ),
        ],
      ),
    );
  }
}

// ============================================================
// GLASS CARD
// ============================================================

class ZAIGlassCard extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry padding;

  const ZAIGlassCard({
    super.key,
    required this.child,
    this.padding = EdgeInsets.zero,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: padding,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(18),
        color: const Color(0xA9081420),
        border: Border.all(
          color: ZAITheme.border.withValues(alpha: .85),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: .22),
            blurRadius: 20,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: child,
    );
  }
}

class _PanelTitle extends StatelessWidget {
  final String title;
  final IconData icon;

  const _PanelTitle({
    required this.title,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Icon(
          icon,
          size: 15,
          color: ZAITheme.cyan,
        ),
        const SizedBox(width: 8),
        Text(
          title,
          style: const TextStyle(
            fontSize: 9,
            color: ZAITheme.muted,
            letterSpacing: 1.3,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }
}

// ============================================================
// ORB ICON
// ============================================================

class ZAIOrbIcon extends StatelessWidget {
  final double size;

  const ZAIOrbIcon({
    super.key,
    required this.size,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        gradient: const RadialGradient(
          colors: [
            Color(0xFF4DFAFF),
            Color(0xFF0077AA),
            Color(0xFF03101A),
          ],
        ),
        boxShadow: [
          BoxShadow(
            color: ZAITheme.cyan.withValues(alpha: .3),
            blurRadius: 16,
          ),
        ],
      ),
      child: Center(
        child: Container(
          width: size * .48,
          height: size * .48,
          decoration: const BoxDecoration(
            shape: BoxShape.circle,
            color: Color(0xFF03101A),
          ),
          child: Center(
            child: Container(
              width: size * .16,
              height: size * .16,
              decoration: const BoxDecoration(
                shape: BoxShape.circle,
                color: ZAITheme.cyan,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ============================================================
// GENERIC PAGE
// ============================================================

class ZAIPage extends StatelessWidget {
  final String title;
  final String subtitle;
  final IconData icon;
  final Widget child;

  const ZAIPage({
    super.key,
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(22),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          ZAIPageHeader(
            title: title,
            subtitle: subtitle,
            icon: icon,
          ),
          const SizedBox(height: 20),
          child,
        ],
      ),
    );
  }
}

// ============================================================
// CHAT
// ============================================================

class ZAIChatPage extends StatefulWidget {
  const ZAIChatPage({super.key});

  @override
  State<ZAIChatPage> createState() => _ZAIChatPageState();
}

class _ZAIChatPageState extends State<ZAIChatPage> {
  final controller = TextEditingController();

  final messages = <Map<String, String>>[
    {
      'role': 'zai',
      'text':
          'ZAI Core online. I am ready for your command.',
    },
  ];

  void send() {
    final text = controller.text.trim();

    if (text.isEmpty) return;

    setState(() {
      messages.add({
        'role': 'user',
        'text': text,
      });

      messages.add({
        'role': 'zai',
        'text':
            'Command received. AI functionality will be connected in the next phase.',
      });

      controller.clear();
    });
  }

  @override
  Widget build(BuildContext context) {
    return ZAIPage(
      title: 'Chat',
      subtitle: 'Direct communication with ZAI',
      icon: Icons.chat_bubble_rounded,
      child: Column(
        children: [
          ZAIGlassCard(
            child: SizedBox(
              height: 500,
              child: Column(
                children: [
                  Expanded(
                    child: ListView.builder(
                      padding: const EdgeInsets.all(18),
                      itemCount: messages.length,
                      itemBuilder: (context, index) {
                        final message = messages[index];

                        final isUser =
                            message['role'] == 'user';

                        return Align(
                          alignment: isUser
                              ? Alignment.centerRight
                              : Alignment.centerLeft,
                          child: Container(
                            constraints:
                                const BoxConstraints(
                              maxWidth: 650,
                            ),
                            margin:
                                const EdgeInsets.only(
                              bottom: 12,
                            ),
                            padding:
                                const EdgeInsets.all(13),
                            decoration: BoxDecoration(
                              borderRadius:
                                  BorderRadius.circular(14),
                              color: isUser
                                  ? ZAITheme.cyan
                                      .withValues(alpha: .08)
                                  : ZAITheme.surface2,
                              border: Border.all(
                                color: isUser
                                    ? ZAITheme.cyan
                                        .withValues(alpha: .18)
                                    : ZAITheme.border,
                              ),
                            ),
                            child: Text(
                              message['text'] ?? '',
                              style: const TextStyle(
                                fontSize: 12,
                                height: 1.5,
                              ),
                            ),
                          ),
                        );
                      },
                    ),
                  ),

                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: const BoxDecoration(
                      border: Border(
                        top: BorderSide(
                          color: ZAITheme.border,
                        ),
                      ),
                    ),
                    child: Row(
                      children: [
                        IconButton(
                          onPressed: () {},
                          icon: const Icon(
                            Icons.attach_file_rounded,
                          ),
                        ),
                        Expanded(
                          child: TextField(
                            controller: controller,
                            onSubmitted: (_) => send(),
                            decoration:
                                const InputDecoration(
                              hintText:
                                  'Enter command for ZAI...',
                              border: InputBorder.none,
                            ),
                          ),
                        ),
                        IconButton(
                          onPressed: send,
                          icon: const Icon(
                            Icons.arrow_upward_rounded,
                            color: ZAITheme.cyan,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ============================================================
// VOICE
// ============================================================

class ZAIVoicePage extends StatefulWidget {
  const ZAIVoicePage({super.key});

  @override
  State<ZAIVoicePage> createState() => _ZAIVoicePageState();
}

class _ZAIVoicePageState extends State<ZAIVoicePage>
    with SingleTickerProviderStateMixin {
  late AnimationController controller;
  bool listening = false;

  @override
  void initState() {
    super.initState();

    controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat();
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ZAIPage(
      title: 'Voice',
      subtitle:
          'Voice interface, wake word and speech control',
      icon: Icons.mic_rounded,
      child: Center(
        child: ZAIGlassCard(
          padding: const EdgeInsets.all(30),
          child: Column(
            children: [
              const SizedBox(height: 20),

              AnimatedBuilder(
                animation: controller,
                builder: (_, child) {
                  return CustomPaint(
                    painter: ZAIVoicePainter(
                      animation: controller.value,
                      active: listening,
                    ),
                    child: SizedBox(
                      width: 300,
                      height: 300,
                      child: Center(
                        child: GestureDetector(
                          onTap: () {
                            setState(() {
                              listening = !listening;
                            });
                          },
                          child: ZAIOrbIcon(
                            size: listening ? 120 : 100,
                          ),
                        ),
                      ),
                    ),
                  );
                },
              ),

              const SizedBox(height: 20),

              Text(
                listening
                    ? 'LISTENING...'
                    : 'READY TO LISTEN',
                style: TextStyle(
                  color: listening
                      ? ZAITheme.cyan
                      : ZAITheme.muted,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 2,
                ),
              ),

              const SizedBox(height: 10),

              const Text(
                'Tap the core to activate voice mode',
                style: TextStyle(
                  color: ZAITheme.muted,
                  fontSize: 11,
                ),
              ),

              const SizedBox(height: 25),

              Wrap(
                spacing: 10,
                children: [
                  _VoiceChip(
                    icon: Icons.mic,
                    label: 'Microphone',
                    active: listening,
                  ),
                  const _VoiceChip(
                    icon: Icons.record_voice_over,
                    label: 'Wake Word',
                    active: false,
                  ),
                  const _VoiceChip(
                    icon: Icons.volume_up,
                    label: 'TTS',
                    active: false,
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _VoiceChip extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool active;

  const _VoiceChip({
    required this.icon,
    required this.label,
    required this.active,
  });

  @override
  Widget build(BuildContext context) {
    return Chip(
      avatar: Icon(
        icon,
        size: 15,
        color: active
            ? ZAITheme.green
            : ZAITheme.muted,
      ),
      label: Text(
        label,
        style: const TextStyle(
          fontSize: 9,
        ),
      ),
      backgroundColor: ZAITheme.surface2,
      side: const BorderSide(
        color: ZAITheme.border,
      ),
    );
  }
}

class ZAIVoicePainter extends CustomPainter {
  final double animation;
  final bool active;

  ZAIVoicePainter({
    required this.animation,
    required this.active,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(
      size.width / 2,
      size.height / 2,
    );

    final maxRadius = size.width * .45;

    for (int i = 0; i < 4; i++) {
      final radius =
          maxRadius * (.35 + i * .17);

      final pulse =
          math.sin(animation * math.pi * 2 + i);

      canvas.drawCircle(
        center,
        radius + pulse * (active ? 7 : 2),
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1
          ..color = ZAITheme.cyan.withValues(
            alpha: active ? .28 : .08,
          ),
      );
    }
  }

  @override
  bool shouldRepaint(covariant ZAIVoicePainter oldDelegate) {
    return true;
  }
}

// ============================================================
// OTHER PAGES
// ============================================================

class ZAIMemoryPage extends StatelessWidget {
  const ZAIMemoryPage({super.key});

  @override
  Widget build(BuildContext context) {
    return const ZAIPage(
      title: 'Memory',
      subtitle: 'Short-term, long-term and semantic memory',
      icon: Icons.memory_rounded,
      child: ZAIPlaceholderGrid(
        items: [
          ('Short-Term Memory', Icons.flash_on),
          ('Long-Term Memory', Icons.storage),
          ('Episodic Memory', Icons.history),
          ('Semantic Memory', Icons.psychology),
          ('Vector Memory', Icons.hub),
          ('Project Memory', Icons.folder_special),
        ],
      ),
    );
  }
}

class ZAIAgentsPage extends StatelessWidget {
  const ZAIAgentsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return const ZAIPage(
      title: 'Agents',
      subtitle: 'Specialized autonomous AI agents',
      icon: Icons.smart_toy_rounded,
      child: ZAIPlaceholderGrid(
        items: [
          ('Computer Agent', Icons.computer),
          ('Browser Agent', Icons.language),
          ('Coding Agent', Icons.code),
          ('Research Agent', Icons.search),
          ('File Agent', Icons.folder),
          ('Automation Agent', Icons.auto_mode),
          ('Android Agent', Icons.android),
          ('Testing Agent', Icons.bug_report),
        ],
      ),
    );
  }
}

class ZAIDevicesPage extends StatelessWidget {
  const ZAIDevicesPage({super.key});

  @override
  Widget build(BuildContext context) {
    return const ZAIPage(
      title: 'Devices',
      subtitle: 'Connected devices and ZAI nodes',
      icon: Icons.devices_rounded,
      child: ZAIPlaceholderGrid(
        items: [
          ('Windows PC', Icons.desktop_windows),
          ('Android', Icons.smartphone),
          ('Laptop', Icons.laptop),
          ('Smart TV', Icons.tv),
          ('Smart Home', Icons.home),
          ('Other Devices', Icons.devices_other),
        ],
      ),
    );
  }
}

class ZAIAutomationPage extends StatelessWidget {
  const ZAIAutomationPage({super.key});

  @override
  Widget build(BuildContext context) {
    return const ZAIPage(
      title: 'Automation',
      subtitle: 'Workflows, routines, triggers and schedules',
      icon: Icons.account_tree_rounded,
      child: ZAIPlaceholderGrid(
        items: [
          ('Workflows', Icons.account_tree),
          ('Schedules', Icons.schedule),
          ('Triggers', Icons.bolt),
          ('Actions', Icons.play_arrow),
          ('Routines', Icons.repeat),
        ],
      ),
    );
  }
}

class ZAIProjectsPage extends StatelessWidget {
  const ZAIProjectsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return const ZAIPage(
      title: 'Projects',
      subtitle: 'Development projects and workspace',
      icon: Icons.folder_rounded,
      child: ZAIPlaceholderGrid(
        items: [
          ('Active Projects', Icons.work),
          ('Completed', Icons.task_alt),
          ('Templates', Icons.dashboard_customize),
          ('Builds', Icons.build),
          ('Git', Icons.source),
        ],
      ),
    );
  }
}

class ZAIKnowledgePage extends StatelessWidget {
  const ZAIKnowledgePage({super.key});

  @override
  Widget build(BuildContext context) {
    return const ZAIPage(
      title: 'Knowledge',
      subtitle: 'Documents, web knowledge and RAG',
      icon: Icons.menu_book_rounded,
      child: ZAIPlaceholderGrid(
        items: [
          ('Documents', Icons.description),
          ('Web Knowledge', Icons.language),
          ('Notes', Icons.note),
          ('Embeddings', Icons.hub),
          ('Indexes', Icons.list_alt),
          ('RAG', Icons.auto_awesome),
        ],
      ),
    );
  }
}

// ============================================================
// EVOLUTION
// ============================================================

class ZAIEvolutionPage extends StatelessWidget {
  const ZAIEvolutionPage({super.key});

  @override
  Widget build(BuildContext context) {
    return ZAIPage(
      title: 'Evolution',
      subtitle:
          'Self-diagnostics, improvement and controlled upgrades',
      icon: Icons.auto_awesome_rounded,
      child: Column(
        children: [
          const ZAIGlassCard(
            padding: EdgeInsets.all(22),
            child: Column(
              children: [
                ZAIOrbIcon(size: 80),
                SizedBox(height: 15),
                Text(
                  'ZAI EVOLUTION ENGINE',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    letterSpacing: 2,
                  ),
                ),
                SizedBox(height: 8),
                Text(
                  'Version 0.1.0',
                  style: TextStyle(
                    color: ZAITheme.cyan,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 16),

          const ZAIPlaceholderGrid(
            items: [
              ('Diagnostics', Icons.health_and_safety),
              ('Self Analysis', Icons.manage_search),
              ('Weakness Detection', Icons.warning_amber),
              ('Improvement Engine', Icons.trending_up),
              ('Benchmark', Icons.speed),
              ('Sandbox', Icons.science),
              ('Version Control', Icons.history),
              ('Rollback', Icons.undo),
            ],
          ),

          const SizedBox(height: 16),

          ZAIGlassCard(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const _PanelTitle(
                  title: 'EVOLUTION PIPELINE',
                  icon: Icons.route,
                ),
                const SizedBox(height: 20),
                _EvolutionStep(
                  number: '01',
                  title: 'Self Observation',
                  active: true,
                ),
                _EvolutionStep(
                  number: '02',
                  title: 'Diagnostics',
                  active: false,
                ),
                _EvolutionStep(
                  number: '03',
                  title: 'Improvement Planning',
                  active: false,
                ),
                _EvolutionStep(
                  number: '04',
                  title: 'Sandbox Testing',
                  active: false,
                ),
                _EvolutionStep(
                  number: '05',
                  title: 'Benchmark & Security',
                  active: false,
                ),
                _EvolutionStep(
                  number: '06',
                  title: 'Controlled Deployment',
                  active: false,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _EvolutionStep extends StatelessWidget {
  final String number;
  final String title;
  final bool active;

  const _EvolutionStep({
    required this.number,
    required this.title,
    required this.active,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: Row(
        children: [
          Container(
            width: 34,
            height: 34,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: active
                  ? ZAITheme.cyan.withValues(alpha: .12)
                  : ZAITheme.surface2,
              border: Border.all(
                color: active
                    ? ZAITheme.cyan.withValues(alpha: .4)
                    : ZAITheme.border,
              ),
            ),
            child: Text(
              number,
              style: TextStyle(
                fontSize: 9,
                color: active
                    ? ZAITheme.cyan
                    : ZAITheme.muted,
              ),
            ),
          ),
          const SizedBox(width: 12),
          Text(
            title,
            style: TextStyle(
              color: active
                  ? ZAITheme.text
                  : ZAITheme.muted,
              fontSize: 11,
            ),
          ),
          const Spacer(),
          if (active)
            const Text(
              'ACTIVE',
              style: TextStyle(
                color: ZAITheme.green,
                fontSize: 8,
                fontWeight: FontWeight.bold,
              ),
            ),
        ],
      ),
    );
  }
}

class ZAIAnalyticsPage extends StatelessWidget {
  const ZAIAnalyticsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return const ZAIPage(
      title: 'Analytics',
      subtitle: 'Performance, usage and system intelligence',
      icon: Icons.analytics_rounded,
      child: ZAIPlaceholderGrid(
        items: [
          ('Performance', Icons.speed),
          ('AI Usage', Icons.psychology),
          ('Task Analytics', Icons.task),
          ('Memory Analytics', Icons.memory),
          ('Agent Analytics', Icons.smart_toy),
          ('System Analytics', Icons.monitor_heart),
        ],
      ),
    );
  }
}

class ZAISettingsPage extends StatelessWidget {
  const ZAISettingsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return const ZAIPage(
      title: 'Settings',
      subtitle: 'Configure your ZAI environment',
      icon: Icons.settings_rounded,
      child: ZAIPlaceholderGrid(
        items: [
          ('General', Icons.tune),
          ('AI Model', Icons.psychology),
          ('Voice', Icons.mic),
          ('Security', Icons.security),
          ('Devices', Icons.devices),
          ('Appearance', Icons.palette),
          ('Notifications', Icons.notifications),
          ('Advanced', Icons.code),
        ],
      ),
    );
  }
}

// ============================================================
// PLACEHOLDER GRID
// ============================================================

class ZAIPlaceholderGrid extends StatelessWidget {
  final List<(String, IconData)> items;

  const ZAIPlaceholderGrid({
    super.key,
    required this.items,
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        int columns = 1;

        if (constraints.maxWidth >= 1200) {
          columns = 4;
        } else if (constraints.maxWidth >= 800) {
          columns = 3;
        } else if (constraints.maxWidth >= 500) {
          columns = 2;
        }

        return GridView.builder(
          itemCount: items.length,
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          gridDelegate:
              SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: columns,
            crossAxisSpacing: 14,
            mainAxisSpacing: 14,
            childAspectRatio: 1.65,
          ),
          itemBuilder: (context, index) {
            final item = items[index];

            return ZAIFeatureCard(
              title: item.$1,
              icon: item.$2,
            );
          },
        );
      },
    );
  }
}

class ZAIFeatureCard extends StatefulWidget {
  final String title;
  final IconData icon;

  const ZAIFeatureCard({
    super.key,
    required this.title,
    required this.icon,
  });

  @override
  State<ZAIFeatureCard> createState() => _ZAIFeatureCardState();
}

class _ZAIFeatureCardState extends State<ZAIFeatureCard> {
  bool hover = false;

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      onEnter: (_) {
        setState(() {
          hover = true;
        });
      },
      onExit: (_) {
        setState(() {
          hover = false;
        });
      },
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        transform: Matrix4.translationValues(
          0,
          hover ? -3 : 0,
          0,
        ),
        child: ZAIGlassCard(
          child: Padding(
            padding: const EdgeInsets.all(17),
            child: Column(
              crossAxisAlignment:
                  CrossAxisAlignment.start,
              children: [
                Container(
                  width: 42,
                  height: 42,
                  decoration: BoxDecoration(
                    borderRadius:
                        BorderRadius.circular(12),
                    color: ZAITheme.cyan.withValues(
                      alpha: hover ? .12 : .05,
                    ),
                    border: Border.all(
                      color: ZAITheme.cyan.withValues(
                        alpha: hover ? .3 : .1,
                      ),
                    ),
                  ),
                  child: Icon(
                    widget.icon,
                    color: ZAITheme.cyan,
                    size: 19,
                  ),
                ),
                const Spacer(),
                Text(
                  widget.title,
                  style: const TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 4),
                const Text(
                  'Ready for integration',
                  style: TextStyle(
                    color: ZAITheme.muted,
                    fontSize: 9,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}