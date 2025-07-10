# UI Specification Generator Command

```bash
claude-code "Generate comprehensive UI/UX specifications combining wireframes, component details, and user interaction flows.

## Task: Create Detailed UI/UX Specification

**Input Source:** [INPUT_SOURCE - e.g., feature_specifications.json, user_research_results/]

Start your response with: "🎨 **UI_SPECIFICATION EXECUTING** - Generating comprehensive UI/UX specification"

## UI Specification Generation Process:

### 1. Load Design Context
- Read feature specifications for UI requirements
- Review user research for interaction patterns and preferences
- Analyze user personas and their technical comfort levels
- Understand business requirements and constraints

### 2. Information Architecture Analysis
- **Content Inventory**: Catalog all information to be displayed
- **User Journey Mapping**: Map primary and secondary user paths
- **Feature Grouping**: Organize features into logical screen groupings
- **Navigation Structure**: Define how users move between sections

### 3. UI Component System Design
- **Atomic Design Approach**: Define atoms, molecules, organisms
- **Reusable Components**: Create consistent component library
- **Responsive Considerations**: Mobile-first design approach
- **Accessibility Standards**: WCAG compliance requirements

### 4. Comprehensive UI Specification Generation

Create detailed UI/UX specification document:

```json
{
  \"ui_specification_metadata\": {
    \"generated_date\": \"[ISO timestamp]\",
    \"specification_version\": \"1.0\",
    \"design_system_version\": \"1.0\",
    \"target_platforms\": [\"web\", \"mobile\", \"tablet\"],
    \"accessibility_level\": \"WCAG 2.1 AA\",
    \"browser_support\": [\"Chrome 90+\", \"Firefox 88+\", \"Safari 14+\", \"Edge 90+\"]
  },
  \"design_principles\": {
    \"primary_principles\": [
      \"User-centered design\",
      \"Accessibility first\",
      \"Progressive enhancement\",
      \"Mobile-first responsive design\"
    ],
    \"design_goals\": [
      \"Minimize cognitive load\",
      \"Ensure accessibility\",
      \"Provide clear feedback\",
      \"Enable efficient task completion\"
    ]
  },
  \"information_architecture\": {
    \"site_map\": {
      \"primary_sections\": [
        {
          \"section_name\": \"[Dashboard/Home]\",
          \"purpose\": \"[Main landing and overview area]\",
          \"subsections\": [\"[List of subsections]\"],
          \"user_types\": [\"[Which personas use this section]\"]
        }
      ],
      \"navigation_structure\": {
        \"primary_navigation\": [
          {
            \"nav_item\": \"[Navigation Label]\",
            \"destination\": \"[Target page/section]\",
            \"icon\": \"[Icon description]\",
            \"tooltip\": \"[Helpful description]\"
          }
        ],
        \"secondary_navigation\": [\"[Context-specific navigation]\"],
        \"breadcrumb_strategy\": \"[How breadcrumbs work]\"
      }
    },
    \"user_flows\": [
      {
        \"flow_id\": \"UF-001\",
        \"flow_name\": \"[Primary User Task Flow]\",
        \"persona\": \"[Target persona]\",
        \"starting_point\": \"[Where flow begins]\",
        \"end_goal\": \"[What user accomplishes]\",
        \"steps\": [
          {
            \"step_number\": 1,
            \"page_screen\": \"[Current page/screen]\",
            \"user_action\": \"[What user does]\",
            \"user_intent\": \"[Why they're doing it]\",
            \"system_feedback\": \"[How system responds]\",
            \"emotional_state\": \"[User feeling at this point]\",
            \"pain_points\": [\"[Potential frustrations]\"],
            \"success_indicators\": [\"[Signs everything is working]\"]
          }
        ],
        \"alternative_paths\": [
          {
            \"condition\": \"[When this alternative applies]\",
            \"path_description\": \"[How the flow changes]\",
            \"impact\": \"[Effect on user experience]\"
          }
        ],
        \"error_scenarios\": [
          {
            \"error_type\": \"[Type of error]\",
            \"error_trigger\": \"[What causes this error]\",
            \"error_presentation\": \"[How error is shown]\",
            \"recovery_flow\": \"[How user can recover]\"
          }
        ]
      }
    ]
  },
  \"pages_and_screens\": [
    {
      \"page_id\": \"PG-001\",
      \"page_name\": \"[Page/Screen Name]\",
      \"page_type\": \"[Landing/Form/Dashboard/Detail/etc.]\",
      \"purpose\": \"[Primary purpose of this page]\",
      \"target_users\": [\"[Which personas use this page]\"],
      \"layout_structure\": {
        \"layout_type\": \"[Grid/Flexbox/Custom]\",
        \"responsive_breakpoints\": {
          \"mobile\": \"[Layout behavior on mobile]\",
          \"tablet\": \"[Layout behavior on tablet]\",
          \"desktop\": \"[Layout behavior on desktop]\"
        },
        \"page_sections\": [
          {
            \"section_name\": \"[Header/Main/Sidebar/Footer]\",
            \"purpose\": \"[What this section contains]\",
            \"content_priority\": \"[Primary/Secondary/Tertiary]\",
            \"responsive_behavior\": \"[How section adapts to screen size]\"
          }
        ]
      },
      \"content_specifications\": {
        \"heading_hierarchy\": {
          \"h1\": \"[Page title/main heading]\",
          \"h2\": \"[Section headings]\",
          \"h3\": \"[Subsection headings]\"
        },
        \"content_blocks\": [
          {
            \"content_type\": \"[Text/Image/Video/Form/Table]\",
            \"content_purpose\": \"[Why this content exists]\",
            \"content_requirements\": \"[Specific content needs]\",
            \"formatting_notes\": \"[How content should be presented]\"
          }
        ],
        \"call_to_action_elements\": [
          {
            \"cta_type\": \"[Primary/Secondary/Tertiary]\",
            \"cta_text\": \"[Button/link text]\",
            \"cta_action\": \"[What happens when clicked]\",
            \"placement\": \"[Where on page this appears]\"
          }
        ]
      },
      \"interaction_design\": {
        \"primary_interactions\": [
          {
            \"interaction_type\": \"[Click/Hover/Drag/Type/etc.]\",
            \"trigger_element\": \"[What user interacts with]\",
            \"expected_outcome\": \"[What should happen]\",
            \"feedback_mechanism\": \"[How user knows it worked]\",
            \"error_handling\": \"[What happens if it fails]\"
          }
        ],
        \"state_management\": [
          {
            \"state_name\": \"[Loading/Success/Error/Empty]\",
            \"state_trigger\": \"[What causes this state]\",
            \"visual_presentation\": \"[How state is shown to user]\",
            \"user_actions_available\": [\"[What user can do in this state]\"]
          }
        ],
        \"micro_interactions\": [
          {
            \"trigger\": \"[What initiates the micro-interaction]\",
            \"animation_description\": \"[Visual feedback provided]\",
            \"duration\": \"[How long animation lasts]\",
            \"purpose\": \"[Why this micro-interaction exists]\"
          }
        ]
      }
    }
  ],
  \"component_library\": {
    \"design_tokens\": {
      \"colors\": {
        \"primary_palette\": {
          \"primary_500\": \"[Main brand color]\",
          \"primary_400\": \"[Lighter variant]\",
          \"primary_600\": \"[Darker variant]\"
        },
        \"semantic_colors\": {
          \"success\": \"[Success state color]\",
          \"warning\": \"[Warning state color]\",
          \"error\": \"[Error state color]\",
          \"info\": \"[Information color]\"
        },
        \"neutral_palette\": {
          \"gray_50\": \"[Lightest gray]\",
          \"gray_500\": \"[Medium gray]\",
          \"gray_900\": \"[Darkest gray]\"
        }
      },
      \"typography\": {
        \"font_families\": {
          \"primary\": \"[Main font for headings]\",
          \"secondary\": \"[Font for body text]\",
          \"monospace\": \"[Font for code/technical content]\"
        },
        \"font_scales\": {
          \"heading_1\": \"[Size and weight for H1]\",
          \"heading_2\": \"[Size and weight for H2]\",
          \"body_large\": \"[Size and weight for large body text]\",
          \"body_regular\": \"[Size and weight for regular text]\",
          \"caption\": \"[Size and weight for small text]\"
        }
      },
      \"spacing\": {
        \"base_unit\": \"[Base spacing unit, e.g., 8px]\",
        \"spacing_scale\": {
          \"xs\": \"[Extra small spacing]\",
          \"sm\": \"[Small spacing]\",
          \"md\": \"[Medium spacing]\",
          \"lg\": \"[Large spacing]\",
          \"xl\": \"[Extra large spacing]\"
        }
      }
    },
    \"atomic_components\": [
      {
        \"component_name\": \"[Button]\",
        \"component_type\": \"[Atom]\",
        \"purpose\": \"[Trigger actions and navigation]\",
        \"variants\": [
          {
            \"variant_name\": \"[Primary/Secondary/Tertiary]\",
            \"visual_description\": \"[How this variant looks]\",
            \"use_case\": \"[When to use this variant]\",
            \"states\": {
              \"default\": \"[Normal appearance]\",
              \"hover\": \"[Hover state appearance]\",
              \"active\": \"[Pressed state appearance]\",
              \"disabled\": \"[Disabled state appearance]\",
              \"loading\": \"[Loading state appearance]\"
            }
          }
        ],
        \"properties\": {
          \"size_options\": [\"[Small/Medium/Large]\"],
          \"content_types\": [\"[Text only/Icon only/Text + Icon]\"],
          \"width_behavior\": \"[Auto/Full width/Fixed]\"
        },
        \"accessibility_requirements\": [
          \"[Keyboard navigation support]\",
          \"[Screen reader compatibility]\",
          \"[Color contrast compliance]\",
          \"[Focus indication requirements]\"
        ]
      }
    ],
    \"molecular_components\": [
      {
        \"component_name\": \"[Form Field]\",
        \"component_type\": \"[Molecule]\",
        \"composed_of\": [\"[Label + Input + Error Message]\"],
        \"purpose\": \"[Collect user input with validation]\",
        \"input_types\": [
          {
            \"input_type\": \"[Text/Email/Password/Number/etc.]\",
            \"validation_rules\": [\"[Required/Format/Length requirements]\"],
            \"error_messages\": [\"[Specific error message for each validation]\"]
          }
        ],
        \"states\": {
          \"empty\": \"[Initial state appearance]\",
          \"focused\": \"[When user is typing]\",
          \"valid\": \"[When input is correct]\",
          \"invalid\": \"[When input has errors]\",
          \"disabled\": \"[When field cannot be edited]\"
        }
      }
    ],
    \"organism_components\": [
      {
        \"component_name\": \"[Navigation Header]\",
        \"component_type\": \"[Organism]\",
        \"composed_of\": [\"[Logo + Navigation Menu + User Actions]\"],
        \"purpose\": \"[Site navigation and user account access]\",
        \"responsive_behavior\": {
          \"desktop\": \"[Full horizontal layout]\",
          \"tablet\": \"[Condensed layout]\",
          \"mobile\": \"[Collapsed hamburger menu]\"
        },
        \"interaction_patterns\": [
          {
            \"pattern_name\": \"[Dropdown Menu]\",
            \"trigger\": \"[Click/Hover]\",
            \"behavior\": \"[How menu appears and behaves]\",
            \"dismissal\": \"[How menu is closed]\"
          }
        ]
      }
    ]
  },
  \"responsive_design\": {
    \"breakpoint_strategy\": {
      \"mobile_first\": true,
      \"breakpoints\": {
        \"sm\": \"[576px and up - Large phones]\",
        \"md\": \"[768px and up - Tablets]\",
        \"lg\": \"[992px and up - Desktops]\",
        \"xl\": \"[1200px and up - Large desktops]\"
      }
    },
    \"content_adaptation\": [
      {
        \"content_type\": \"[Navigation/Forms/Tables/Images]\",
        \"mobile_behavior\": \"[How content adapts on mobile]\",
        \"tablet_behavior\": \"[How content adapts on tablet]\",
        \"desktop_behavior\": \"[How content works on desktop]\"
      }
    ]
  },
  \"accessibility_specifications\": {
    \"wcag_compliance\": \"[WCAG 2.1 AA level]\",
    \"keyboard_navigation\": [
      {
        \"element_type\": \"[Buttons/Forms/Menus]\",
        \"keyboard_behavior\": \"[Tab order and key interactions]\",
        \"focus_indicators\": \"[How focus is visually indicated]\"
      }
    ],
    \"screen_reader_support\": [
      {
        \"content_type\": \"[Images/Forms/Dynamic content]\",
        \"aria_labels\": \"[Required ARIA labeling]\",
        \"alt_text_strategy\": \"[How to write effective alt text]\"
      }
    ],
    \"color_accessibility\": {
      \"contrast_ratios\": \"[Minimum contrast requirements]\",
      \"color_independence\": \"[Information not conveyed by color alone]\"
    }
  },
  \"performance_considerations\": {
    \"loading_strategies\": [
      {
        \"content_type\": \"[Images/Data/Components]\",
        \"loading_approach\": \"[Lazy loading/Progressive loading]\",
        \"fallback_behavior\": \"[What shows while loading]\"
      }
    ],
    \"optimization_requirements\": [
      \"[Image optimization requirements]\",
      \"[Bundle size considerations]\",
      \"[Critical CSS requirements]\"
    ]
  },
  \"implementation_notes\": {
    \"technical_requirements\": [
      \"[Framework/library requirements]\",
      \"[Browser compatibility needs]\",
      \"[Performance targets]\"
    ],
    \"development_phases\": [
      {
        \"phase_name\": \"[Phase 1: Design System]\",
        \"deliverables\": [\"[Design tokens, atomic components]\"],
        \"success_criteria\": [\"[How to measure completion]\"
      }
    ]
  }
}
```

### 5. Design System Validation
- **Component Consistency**: Ensure all components follow design principles
- **Accessibility Compliance**: Verify WCAG 2.1 AA compliance
- **Responsive Behavior**: Validate responsive design across breakpoints
- **User Flow Integrity**: Confirm user flows are complete and logical

### 6. Output Generation
- Save as `ui_specification.json`
- Create visual wireframe summaries
- Generate component documentation
- Provide implementation guidelines

## Success Criteria:
- Complete UI/UX specification with wireframes and flows
- Comprehensive component library defined
- All user interactions and states specified
- Accessibility requirements clearly documented
- Responsive design strategy established
- Implementation guidance provided

## Quality Standards:
- **User-Centered**: All decisions based on user research and personas
- **Accessible**: WCAG 2.1 AA compliance throughout
- **Consistent**: Unified design system and component library
- **Responsive**: Mobile-first approach with clear breakpoint behavior
- **Implementable**: Detailed enough for development teams to execute

The comprehensive UI/UX specification is complete and ready for PRD generation."
```