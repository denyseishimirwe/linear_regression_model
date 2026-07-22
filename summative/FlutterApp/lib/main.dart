import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const ExamScorePredictorApp());
}

class ExamScorePredictorApp extends StatelessWidget {
  const ExamScorePredictorApp({super.key});

  @override
  Widget build(BuildContext context) {
    const teal = Color(0xFF0F6E56);
    return MaterialApp(
      title: 'Exam Score Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: teal),
        scaffoldBackgroundColor: const Color(0xFFF7F8F7),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(10),
            borderSide: const BorderSide(color: Color(0xFFDDE1DE)),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(10),
            borderSide: const BorderSide(color: Color(0xFFDDE1DE)),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(10),
            borderSide: const BorderSide(color: teal, width: 1.5),
          ),
          errorBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(10),
            borderSide: const BorderSide(color: Colors.redAccent),
          ),
        ),
      ),
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final _formKey = GlobalKey<FormState>();

  // IMPORTANT: replace with your actual deployed Render URL once live.
  static const String apiBaseUrl = 'https://student-exam-predictor-ae37.onrender.com';

  // Text controllers - one per model input variable (13 total)
  final studyHoursController = TextEditingController();
  final attendanceController = TextEditingController();
  final socialMediaController = TextEditingController();
  final netflixController = TextEditingController();
  final sleepController = TextEditingController();
  final exerciseController = TextEditingController();
  final mentalHealthController = TextEditingController();
  final dietQualityController = TextEditingController();
  final internetQualityController = TextEditingController();
  final genderController = TextEditingController();
  final partTimeJobController = TextEditingController();
  final extracurricularController = TextEditingController();
  final parentalEducationController = TextEditingController();

  bool _isLoading = false;
  String? _resultText;
  bool _isError = false;

  @override
  void dispose() {
    for (final c in [
      studyHoursController,
      attendanceController,
      socialMediaController,
      netflixController,
      sleepController,
      exerciseController,
      mentalHealthController,
      dietQualityController,
      internetQualityController,
      genderController,
      partTimeJobController,
      extracurricularController,
      parentalEducationController,
    ]) {
      c.dispose();
    }
    super.dispose();
  }

  // ---- Validators (mirror the API's Pydantic range constraints) ----
  String? _numberValidator(String? value, {required double min, required double max, String? label}) {
    if (value == null || value.trim().isEmpty) return 'Required';
    final parsed = double.tryParse(value.trim());
    if (parsed == null) return 'Enter a number';
    if (parsed < min || parsed > max) return 'Must be $min-$max';
    return null;
  }

  String? _choiceValidator(String? value, List<String> allowed) {
    if (value == null || value.trim().isEmpty) return 'Required';
    final match = allowed.firstWhere(
      (a) => a.toLowerCase() == value.trim().toLowerCase(),
      orElse: () => '',
    );
    if (match.isEmpty) return 'Must be one of: ${allowed.join(", ")}';
    return null;
  }

  // Normalizes a typed value ("good") to match the exact case the API expects ("Good")
  String _normalizeChoice(String value, List<String> allowed) {
    return allowed.firstWhere(
      (a) => a.toLowerCase() == value.trim().toLowerCase(),
      orElse: () => value.trim(),
    );
  }

  Future<void> _submitPrediction() async {
    setState(() {
      _resultText = null;
      _isError = false;
    });

    if (!_formKey.currentState!.validate()) {
      setState(() {
        _isError = true;
        _resultText = 'Please fix the highlighted fields before predicting.';
      });
      return;
    }

    setState(() => _isLoading = true);

    final payload = {
      'study_hours_per_day': double.parse(studyHoursController.text.trim()),
      'social_media_hours': double.parse(socialMediaController.text.trim()),
      'netflix_hours': double.parse(netflixController.text.trim()),
      'attendance_percentage': double.parse(attendanceController.text.trim()),
      'sleep_hours': double.parse(sleepController.text.trim()),
      'exercise_frequency': int.parse(exerciseController.text.trim()),
      'mental_health_rating': int.parse(mentalHealthController.text.trim()),
      'diet_quality': _normalizeChoice(dietQualityController.text, ['Poor', 'Fair', 'Good']),
      'internet_quality': _normalizeChoice(internetQualityController.text, ['Poor', 'Average', 'Good']),
      'gender': _normalizeChoice(genderController.text, ['Male', 'Female', 'Other']),
      'part_time_job': _normalizeChoice(partTimeJobController.text, ['Yes', 'No']),
      'extracurricular_participation': _normalizeChoice(extracurricularController.text, ['Yes', 'No']),
      'parental_education_level': _normalizeChoice(
          parentalEducationController.text, ['High School', 'Bachelor', 'Master', 'Unknown']),
    };

    try {
      final response = await http.post(
        Uri.parse('$apiBaseUrl/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _isError = false;
          _resultText = data['predicted_exam_score'].toString();
        });
      } else {
        final data = jsonDecode(response.body);
        setState(() {
          _isError = true;
          _resultText = 'Error: ${data['detail'] ?? 'Could not get a prediction.'}';
        });
      }
    } catch (e) {
      setState(() {
        _isError = true;
        _resultText = 'Could not reach the server. Check your internet connection and try again.';
      });
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    const teal = Color(0xFF0F6E56);

    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(6),
                      decoration: BoxDecoration(
                        color: teal.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: const Icon(Icons.school, color: teal, size: 20),
                    ),
                    const SizedBox(width: 8),
                    const Text(
                      'Exam Score Predictor',
                      style: TextStyle(fontSize: 19, fontWeight: FontWeight.w700),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                const Text(
                  'Predict your exam score from your habits and lifestyle.',
                  style: TextStyle(fontSize: 13, color: Colors.black54),
                ),
                const SizedBox(height: 20),

                _SectionCard(
                  icon: Icons.menu_book_outlined,
                  title: 'Study habits',
                  children: [
                    _buildField('Study hours/day', 'e.g. 5 (0-12)', studyHoursController,
                        validator: (v) => _numberValidator(v, min: 0, max: 12)),
                    _buildField('Attendance %', 'e.g. 85 (0-100)', attendanceController,
                        validator: (v) => _numberValidator(v, min: 0, max: 100)),
                    _buildField('Social media (hrs)', 'e.g. 2 (0-12)', socialMediaController,
                        validator: (v) => _numberValidator(v, min: 0, max: 12)),
                    _buildField('Netflix (hrs)', 'e.g. 1 (0-12)', netflixController,
                        validator: (v) => _numberValidator(v, min: 0, max: 12)),
                  ],
                ),

                _SectionCard(
                  icon: Icons.favorite_border,
                  title: 'Wellbeing',
                  children: [
                    _buildField('Sleep (hrs)', 'e.g. 7 (0-24)', sleepController,
                        validator: (v) => _numberValidator(v, min: 0, max: 24)),
                    _buildField('Exercise/week', 'e.g. 3 (0-14)', exerciseController,
                        validator: (v) => _numberValidator(v, min: 0, max: 14)),
                    _buildField('Mental health (1-10)', 'e.g. 7', mentalHealthController,
                        validator: (v) => _numberValidator(v, min: 1, max: 10)),
                    _buildField('Diet quality', 'Poor, Fair, or Good', dietQualityController,
                        validator: (v) => _choiceValidator(v, ['Poor', 'Fair', 'Good'])),
                    _buildField('Internet quality', 'Poor, Average, or Good', internetQualityController,
                        validator: (v) => _choiceValidator(v, ['Poor', 'Average', 'Good'])),
                  ],
                ),

                _SectionCard(
                  icon: Icons.badge_outlined,
                  title: 'Background',
                  children: [
                    _buildField('Gender', 'Male, Female, or Other', genderController,
                        validator: (v) => _choiceValidator(v, ['Male', 'Female', 'Other'])),
                    _buildField('Part-time job', 'Yes or No', partTimeJobController,
                        validator: (v) => _choiceValidator(v, ['Yes', 'No'])),
                    _buildField('Extracurriculars', 'Yes or No', extracurricularController,
                        validator: (v) => _choiceValidator(v, ['Yes', 'No'])),
                    _buildField('Parental education', 'High School, Bachelor, Master, or Unknown',
                        parentalEducationController,
                        validator: (v) =>
                            _choiceValidator(v, ['High School', 'Bachelor', 'Master', 'Unknown'])),
                  ],
                ),

                const SizedBox(height: 8),
                SizedBox(
                  width: double.infinity,
                  height: 50,
                  child: ElevatedButton.icon(
                    onPressed: _isLoading ? null : _submitPrediction,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: teal,
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    ),
                    icon: _isLoading
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                          )
                        : const Icon(Icons.auto_awesome, size: 18),
                    label: Text(_isLoading ? 'Predicting...' : 'Predict Score',
                        style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 15)),
                  ),
                ),

                if (_resultText != null) ...[
                  const SizedBox(height: 16),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 14),
                    decoration: BoxDecoration(
                      color: _isError ? const Color(0xFFFDECEC) : teal.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: _isError ? Colors.redAccent.withOpacity(0.4) : teal.withOpacity(0.3),
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          _isError ? 'Error' : 'Predicted exam score',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            color: _isError ? Colors.redAccent : teal,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          _isError ? _resultText! : _resultText!,
                          style: TextStyle(
                            fontSize: _isError ? 14 : 26,
                            fontWeight: FontWeight.w700,
                            color: _isError ? Colors.redAccent : teal,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildField(
    String label,
    String hint,
    TextEditingController controller, {
    required String? Function(String?) validator,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: const TextStyle(fontSize: 12.5, fontWeight: FontWeight.w600)),
          const SizedBox(height: 6),
          TextFormField(
            controller: controller,
            decoration: InputDecoration(hintText: hint, hintStyle: const TextStyle(fontSize: 13)),
            style: const TextStyle(fontSize: 14),
            validator: validator,
          ),
        ],
      ),
    );
  }
}

class _SectionCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final List<Widget> children;

  const _SectionCard({required this.icon, required this.title, required this.children});

  @override
  Widget build(BuildContext context) {
    const teal = Color(0xFF0F6E56);
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFE7E9E7)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, size: 18, color: teal),
              const SizedBox(width: 6),
              Text(title, style: const TextStyle(fontSize: 14.5, fontWeight: FontWeight.w700)),
            ],
          ),
          const SizedBox(height: 12),
          ...children,
        ],
      ),
    );
  }
}
